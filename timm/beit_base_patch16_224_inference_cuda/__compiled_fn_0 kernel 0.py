
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


# kernel path: /tmp/torchinductor_youkaichao/qr/cqry3nzs5l2gnfrcnujjhy76hbsdcxlki3aj2hfoxfbb3p6qty7c.py
# Source Nodes: [cat_25, x_6], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_25 => cat
# x_6 => var_mean
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9456
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 197
    x1 = (xindex // 197) % 6
    x4 = (xindex // 197)
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
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
        tmp9 = tl.full([1, 1], 197, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((196*r3) + (25088*x4) + (((-1) + x0) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r3 + (128*x1)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp8, tmp13, tmp14)
        tmp16 = tl.where(tmp4, tmp7, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight,
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp18, xmask)
    tl.store(out_ptr1 + (x5), tmp19, xmask)
    tl.store(out_ptr2 + (x5), tmp20, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/lx/clxbm6zuzevkwguh4mfktivj7rawgaqchbtxyqlghdagmrgporjx.py
# Source Nodes: [cat_25, x_6], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_25 => cat
# x_6 => var_mean
triton_per_fused_cat_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 197
    x1 = (xindex // 197)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (197*r2) + (1182*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (197*r2) + (1182*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (197*r2) + (1182*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ui/cui42ffvwiirr5kl2szmhpq7hb4oulfdk7mwbiyag4k2rs62qm5r.py
# Source Nodes: [cat_25, x_6], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_25 => cat
# x_6 => add, add_1, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_cat_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768) % 197
    x0 = xindex % 768
    x2 = (xindex // 151296)
    x3 = (xindex // 768)
    x4 = xindex
    tmp17 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = 768.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp18 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jv6nctpzmj57gbyves5epptcamlxcnrwokrjdyk6cdotbsfstb.py
# Source Nodes: [cat_24], Original ATen: [aten.cat]
# cat_24 => cat_1
triton_poi_fused_cat_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 1536, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-768) + x0), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 2304, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-1536) + x0), tmp15 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/cebhq4y2km7qgrs62yn3ccj6crydzr4s5ksmfjbwerwlsphchut4.py
# Source Nodes: [x_7], Original ATen: [aten.constant_pad_nd]
# x_7 => constant_pad_nd
triton_poi_fused_constant_pad_nd_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 472800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200
    x1 = (xindex // 200) % 197
    x2 = (xindex // 39400)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 197, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (197*x1)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp3 + 732
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 732)) | ~(xmask & tmp2), "index out of bounds: 0 <= tmp6 < 732")
    tmp7 = tl.load(in_ptr1 + (x2 + (12*tmp6)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5p/c5pecfoasltu7zkdluxs2us3q4byyjb6u43dpjocvm3nycliluzc.py
# Source Nodes: [cat_25, mul, x_11, x_12], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm]
# cat_25 => cat
# mul => mul_2
# x_11 => add_2
# x_12 => add_3, add_4, mul_3, mul_4, rsqrt_1, sub_1, var_mean_1
triton_per_fused_add_cat_mul_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_mul_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
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
    tmp17 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 * tmp20
    tmp22 = tmp16 + tmp21
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
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czd5rypf6lpa3t3wlnwvhhj4zjmc5zcjo6hipyqvwkpeooapub7h.py
# Source Nodes: [x_14], Original ATen: [aten.gelu]
# x_14 => add_5, erf, mul_5, mul_6, mul_7
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
    xnumel = 4841472
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


# kernel path: /tmp/torchinductor_youkaichao/fm/cfm6yz2smjptzv6y6kvwgtudcenbyo72vz7m4p6isugrnnykd5ux.py
# Source Nodes: [mul_1, x_20, x_21], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# mul_1 => mul_8
# x_20 => add_6
# x_21 => add_7, add_8, mul_10, mul_9, rsqrt_2, sub_2, var_mean_2
triton_per_fused_add_mul_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fc/cfcerc54lhqe37s6jam6ue5jndlsqrcq3jhb6qcqea57w6a6ttmo.py
# Source Nodes: [mul_1, mul_2, x_20, x_26, x_27], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# mul_1 => mul_8
# mul_2 => mul_11
# x_20 => add_6
# x_26 => add_9
# x_27 => add_10, add_11, mul_12, mul_13, rsqrt_3, sub_3, var_mean_3
triton_per_fused_add_mul_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp12, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrts25scamndbtpfcaib4iynic6rz7se5ncq2tgykpxholnxsis.py
# Source Nodes: [x_188], Original ATen: [aten.mean]
# x_188 => mean
triton_red_fused_mean_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768) % 2
    x2 = (xindex // 1536)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (768 + x0 + (768*r3) + (75264*x1) + (151296*x2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (768 + x0 + (768*r3) + (75264*x1) + (151296*x2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp1 * tmp4
        tmp6 = tmp0 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5e/c5eyfaszlaziaeo56qsuqalunae2yeey6igknlgkkndbawgw4pz4.py
# Source Nodes: [x_188], Original ATen: [aten.mean]
# x_188 => mean
triton_per_fused_mean_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/cllfanenlbrahhgk7u6zikn5gb66zle2nlyxrbxyp5cwgltciqkm.py
# Source Nodes: [x_188, x_190], Original ATen: [aten.mean, aten.native_layer_norm]
# x_188 => mean
# x_190 => add_84, add_85, mul_108, mul_109, rsqrt_24, sub_24, var_mean_24
triton_per_fused_mean_native_layer_norm_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 8
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
    tmp26 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 196.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 768.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (2304, 768), (768, 1))
    assert_size_stride(arg7_1, (732, 12), (12, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (2304, 768), (768, 1))
    assert_size_stride(arg17_1, (732, 12), (12, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (2304, 768), (768, 1))
    assert_size_stride(arg27_1, (732, 12), (12, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (2304, 768), (768, 1))
    assert_size_stride(arg37_1, (732, 12), (12, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (2304, 768), (768, 1))
    assert_size_stride(arg47_1, (732, 12), (12, 1))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (2304, 768), (768, 1))
    assert_size_stride(arg57_1, (732, 12), (12, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (2304, 768), (768, 1))
    assert_size_stride(arg67_1, (732, 12), (12, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (2304, 768), (768, 1))
    assert_size_stride(arg77_1, (732, 12), (12, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (2304, 768), (768, 1))
    assert_size_stride(arg87_1, (732, 12), (12, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (2304, 768), (768, 1))
    assert_size_stride(arg97_1, (732, 12), (12, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (2304, 768), (768, 1))
    assert_size_stride(arg107_1, (732, 12), (12, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (2304, 768), (768, 1))
    assert_size_stride(arg117_1, (732, 12), (12, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, 768), (768, 1))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (3072, 768), (768, 1))
    assert_size_stride(arg128_1, (3072, ), (1, ))
    assert_size_stride(arg129_1, (768, 3072), (3072, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, 768), (768, 1))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (3072, 768), (768, 1))
    assert_size_stride(arg134_1, (3072, ), (1, ))
    assert_size_stride(arg135_1, (768, 3072), (3072, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (3072, 768), (768, 1))
    assert_size_stride(arg140_1, (3072, ), (1, ))
    assert_size_stride(arg141_1, (768, 3072), (3072, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, 768), (768, 1))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (3072, 768), (768, 1))
    assert_size_stride(arg146_1, (3072, ), (1, ))
    assert_size_stride(arg147_1, (768, 3072), (3072, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, 768), (768, 1))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (3072, 768), (768, 1))
    assert_size_stride(arg152_1, (3072, ), (1, ))
    assert_size_stride(arg153_1, (768, 3072), (3072, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (3072, 768), (768, 1))
    assert_size_stride(arg158_1, (3072, ), (1, ))
    assert_size_stride(arg159_1, (768, 3072), (3072, 1))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, 768), (768, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (3072, 768), (768, 1))
    assert_size_stride(arg164_1, (3072, ), (1, ))
    assert_size_stride(arg165_1, (768, 3072), (3072, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, 768), (768, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (3072, 768), (768, 1))
    assert_size_stride(arg170_1, (3072, ), (1, ))
    assert_size_stride(arg171_1, (768, 3072), (3072, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, 768), (768, 1))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (3072, 768), (768, 1))
    assert_size_stride(arg176_1, (3072, ), (1, ))
    assert_size_stride(arg177_1, (768, 3072), (3072, 1))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, 768), (768, 1))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (3072, 768), (768, 1))
    assert_size_stride(arg182_1, (3072, ), (1, ))
    assert_size_stride(arg183_1, (768, 3072), (3072, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 768), (768, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (3072, 768), (768, 1))
    assert_size_stride(arg188_1, (3072, ), (1, ))
    assert_size_stride(arg189_1, (768, 3072), (3072, 1))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, 768), (768, 1))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (3072, 768), (768, 1))
    assert_size_stride(arg194_1, (3072, ), (1, ))
    assert_size_stride(arg195_1, (768, 3072), (3072, 1))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (1000, 768), (768, 1))
    assert_size_stride(arg198_1, (1000, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (197, 197), (197, 1))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (197, 197), (197, 1))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (197, 197), (197, 1))
    assert_size_stride(arg205_1, (768, ), (1, ))
    assert_size_stride(arg206_1, (197, 197), (197, 1))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (197, 197), (197, 1))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (197, 197), (197, 1))
    assert_size_stride(arg211_1, (768, ), (1, ))
    assert_size_stride(arg212_1, (197, 197), (197, 1))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (197, 197), (197, 1))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (197, 197), (197, 1))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (197, 197), (197, 1))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (197, 197), (197, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (197, 197), (197, 1))
    assert_size_stride(arg223_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg223_1, arg123_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg123_1
        del arg223_1
        buf1 = empty_strided((8, 197, 1, 6), (1182, 1, 9456, 197), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 197, 1, 6), (1182, 1, 9456, 197), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 197, 1, 6), (1182, 1, 9456, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25, x_6], Original ATen: [aten.cat, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_cat_native_layer_norm_0.run(arg0_1, buf0, arg124_1, buf1, buf2, buf3, 9456, 128, grid=grid(9456), stream=stream0)
        buf4 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25, x_6], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1576, 6, grid=grid(1576), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25, x_6], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_2.run(arg0_1, buf0, arg124_1, buf4, buf5, arg2_1, arg3_1, buf7, 1210368, grid=grid(1210368), stream=stream0)
        del arg2_1
        del arg3_1
        del buf4
        del buf5
        buf8 = empty((2304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg4_1, arg199_1, arg5_1, buf8, 2304, grid=grid(2304), stream=stream0)
        del arg199_1
        del arg4_1
        del arg5_1
        buf9 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24, qkv], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf8, reinterpret_tensor(buf7, (1576, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del arg6_1
        buf10 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg200_1, arg7_1, buf10, 472800, grid=grid(472800), stream=stream0)
        del arg200_1
        del arg7_1
        # Source Nodes: [x_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf11 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf10, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf12 = buf11[0]
        del buf11
        buf16 = reinterpret_tensor(buf7, (1576, 768), (768, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf12, (1576, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 768), (1, 768), 0), out=buf16)
        del arg125_1
        buf17 = reinterpret_tensor(buf16, (8, 197, 768), (151296, 768, 1), 0); del buf16  # reuse
        buf21 = reinterpret_tensor(buf12, (8, 197, 768), (151296, 768, 1), 0); del buf12  # reuse
        # Source Nodes: [cat_25, mul, x_11, x_12], Original ATen: [aten.add, aten.cat, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_cat_mul_native_layer_norm_5.run(buf17, arg0_1, buf0, arg124_1, arg1_1, arg126_1, arg9_1, arg10_1, buf21, 1576, 768, grid=grid(1576), stream=stream0)
        del arg0_1
        del arg10_1
        del arg124_1
        del arg126_1
        del arg1_1
        del arg9_1
        del buf0
        buf22 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (1576, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 3072), (1, 768), 0), out=buf22)
        del arg127_1
        buf23 = reinterpret_tensor(buf22, (8, 197, 3072), (605184, 3072, 1), 0); del buf22  # reuse
        # Source Nodes: [x_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf23, arg128_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg128_1
        buf24 = reinterpret_tensor(buf21, (1576, 768), (768, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg129_1, (3072, 768), (1, 3072), 0), out=buf24)
        del arg129_1
        buf28 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_1, x_20, x_21], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf17, arg8_1, buf24, arg130_1, arg12_1, arg13_1, buf28, 1576, 768, grid=grid(1576), stream=stream0)
        del arg12_1
        del arg13_1
        buf29 = buf8; del buf8  # reuse
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg14_1, arg201_1, arg15_1, buf29, 2304, grid=grid(2304), stream=stream0)
        del arg14_1
        del arg15_1
        del arg201_1
        buf30 = buf9; del buf9  # reuse
        # Source Nodes: [cat_23, qkv_2], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf29, reinterpret_tensor(buf28, (1576, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf30)
        del arg16_1
        buf31 = buf10; del buf10  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg202_1, arg17_1, buf31, 472800, grid=grid(472800), stream=stream0)
        del arg17_1
        del arg202_1
        # Source Nodes: [x_22], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf32 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf31, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf33 = buf32[0]
        del buf32
        buf37 = reinterpret_tensor(buf28, (1576, 768), (768, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (1576, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 768), (1, 768), 0), out=buf37)
        del arg131_1
        buf38 = reinterpret_tensor(buf37, (8, 197, 768), (151296, 768, 1), 0); del buf37  # reuse
        buf42 = reinterpret_tensor(buf33, (8, 197, 768), (151296, 768, 1), 0); del buf33  # reuse
        # Source Nodes: [mul_1, mul_2, x_20, x_26, x_27], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf38, buf17, arg8_1, buf24, arg130_1, arg11_1, arg132_1, arg19_1, arg20_1, buf42, 1576, 768, grid=grid(1576), stream=stream0)
        del arg11_1
        del arg130_1
        del arg132_1
        del arg19_1
        del arg20_1
        del arg8_1
        del buf17
        buf43 = reinterpret_tensor(buf23, (1576, 3072), (3072, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (1576, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 3072), (1, 768), 0), out=buf43)
        del arg133_1
        buf44 = reinterpret_tensor(buf43, (8, 197, 3072), (605184, 3072, 1), 0); del buf43  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf44, arg134_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg134_1
        buf45 = reinterpret_tensor(buf42, (1576, 768), (768, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg135_1, (3072, 768), (1, 3072), 0), out=buf45)
        del arg135_1
        buf49 = reinterpret_tensor(buf24, (8, 197, 768), (151296, 768, 1), 0); del buf24  # reuse
        # Source Nodes: [mul_3, x_35, x_36], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf38, arg18_1, buf45, arg136_1, arg22_1, arg23_1, buf49, 1576, 768, grid=grid(1576), stream=stream0)
        del arg22_1
        del arg23_1
        buf50 = buf29; del buf29  # reuse
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg24_1, arg203_1, arg25_1, buf50, 2304, grid=grid(2304), stream=stream0)
        del arg203_1
        del arg24_1
        del arg25_1
        buf51 = buf30; del buf30  # reuse
        # Source Nodes: [cat_22, qkv_4], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf50, reinterpret_tensor(buf49, (1576, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf51)
        del arg26_1
        buf52 = buf31; del buf31  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg204_1, arg27_1, buf52, 472800, grid=grid(472800), stream=stream0)
        del arg204_1
        del arg27_1
        # Source Nodes: [x_37], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf53 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf52, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf54 = buf53[0]
        del buf53
        buf58 = reinterpret_tensor(buf49, (1576, 768), (768, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (1576, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), out=buf58)
        del arg137_1
        buf59 = reinterpret_tensor(buf58, (8, 197, 768), (151296, 768, 1), 0); del buf58  # reuse
        buf63 = reinterpret_tensor(buf54, (8, 197, 768), (151296, 768, 1), 0); del buf54  # reuse
        # Source Nodes: [mul_3, mul_4, x_35, x_41, x_42], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf59, buf38, arg18_1, buf45, arg136_1, arg21_1, arg138_1, arg29_1, arg30_1, buf63, 1576, 768, grid=grid(1576), stream=stream0)
        del arg136_1
        del arg138_1
        del arg18_1
        del arg21_1
        del arg29_1
        del arg30_1
        del buf38
        buf64 = reinterpret_tensor(buf44, (1576, 3072), (3072, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (1576, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 3072), (1, 768), 0), out=buf64)
        del arg139_1
        buf65 = reinterpret_tensor(buf64, (8, 197, 3072), (605184, 3072, 1), 0); del buf64  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf65, arg140_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg140_1
        buf66 = reinterpret_tensor(buf63, (1576, 768), (768, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg141_1, (3072, 768), (1, 3072), 0), out=buf66)
        del arg141_1
        buf70 = reinterpret_tensor(buf45, (8, 197, 768), (151296, 768, 1), 0); del buf45  # reuse
        # Source Nodes: [mul_5, x_50, x_51], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf59, arg28_1, buf66, arg142_1, arg32_1, arg33_1, buf70, 1576, 768, grid=grid(1576), stream=stream0)
        del arg32_1
        del arg33_1
        buf71 = buf50; del buf50  # reuse
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg34_1, arg205_1, arg35_1, buf71, 2304, grid=grid(2304), stream=stream0)
        del arg205_1
        del arg34_1
        del arg35_1
        buf72 = buf51; del buf51  # reuse
        # Source Nodes: [cat_21, qkv_6], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf71, reinterpret_tensor(buf70, (1576, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf72)
        del arg36_1
        buf73 = buf52; del buf52  # reuse
        # Source Nodes: [x_52], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg206_1, arg37_1, buf73, 472800, grid=grid(472800), stream=stream0)
        del arg206_1
        del arg37_1
        # Source Nodes: [x_52], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf74 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf73, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf75 = buf74[0]
        del buf74
        buf79 = reinterpret_tensor(buf70, (1576, 768), (768, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1576, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 768), (1, 768), 0), out=buf79)
        del arg143_1
        buf80 = reinterpret_tensor(buf79, (8, 197, 768), (151296, 768, 1), 0); del buf79  # reuse
        buf84 = reinterpret_tensor(buf75, (8, 197, 768), (151296, 768, 1), 0); del buf75  # reuse
        # Source Nodes: [mul_5, mul_6, x_50, x_56, x_57], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf80, buf59, arg28_1, buf66, arg142_1, arg31_1, arg144_1, arg39_1, arg40_1, buf84, 1576, 768, grid=grid(1576), stream=stream0)
        del arg142_1
        del arg144_1
        del arg28_1
        del arg31_1
        del arg39_1
        del arg40_1
        del buf59
        buf85 = reinterpret_tensor(buf65, (1576, 3072), (3072, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (1576, 768), (768, 1), 0), reinterpret_tensor(arg145_1, (768, 3072), (1, 768), 0), out=buf85)
        del arg145_1
        buf86 = reinterpret_tensor(buf85, (8, 197, 3072), (605184, 3072, 1), 0); del buf85  # reuse
        # Source Nodes: [x_59], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf86, arg146_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg146_1
        buf87 = reinterpret_tensor(buf84, (1576, 768), (768, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf86, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg147_1, (3072, 768), (1, 3072), 0), out=buf87)
        del arg147_1
        buf91 = reinterpret_tensor(buf66, (8, 197, 768), (151296, 768, 1), 0); del buf66  # reuse
        # Source Nodes: [mul_7, x_65, x_66], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf80, arg38_1, buf87, arg148_1, arg42_1, arg43_1, buf91, 1576, 768, grid=grid(1576), stream=stream0)
        del arg42_1
        del arg43_1
        buf92 = buf71; del buf71  # reuse
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg44_1, arg207_1, arg45_1, buf92, 2304, grid=grid(2304), stream=stream0)
        del arg207_1
        del arg44_1
        del arg45_1
        buf93 = buf72; del buf72  # reuse
        # Source Nodes: [cat_20, qkv_8], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf92, reinterpret_tensor(buf91, (1576, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf93)
        del arg46_1
        buf94 = buf73; del buf73  # reuse
        # Source Nodes: [x_67], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg208_1, arg47_1, buf94, 472800, grid=grid(472800), stream=stream0)
        del arg208_1
        del arg47_1
        # Source Nodes: [x_67], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf95 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf94, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf96 = buf95[0]
        del buf95
        buf100 = reinterpret_tensor(buf91, (1576, 768), (768, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (1576, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 768), (1, 768), 0), out=buf100)
        del arg149_1
        buf101 = reinterpret_tensor(buf100, (8, 197, 768), (151296, 768, 1), 0); del buf100  # reuse
        buf105 = reinterpret_tensor(buf96, (8, 197, 768), (151296, 768, 1), 0); del buf96  # reuse
        # Source Nodes: [mul_7, mul_8, x_65, x_71, x_72], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf101, buf80, arg38_1, buf87, arg148_1, arg41_1, arg150_1, arg49_1, arg50_1, buf105, 1576, 768, grid=grid(1576), stream=stream0)
        del arg148_1
        del arg150_1
        del arg38_1
        del arg41_1
        del arg49_1
        del arg50_1
        del buf80
        buf106 = reinterpret_tensor(buf86, (1576, 3072), (3072, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (1576, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 3072), (1, 768), 0), out=buf106)
        del arg151_1
        buf107 = reinterpret_tensor(buf106, (8, 197, 3072), (605184, 3072, 1), 0); del buf106  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf107, arg152_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg152_1
        buf108 = reinterpret_tensor(buf105, (1576, 768), (768, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg153_1, (3072, 768), (1, 3072), 0), out=buf108)
        del arg153_1
        buf112 = reinterpret_tensor(buf87, (8, 197, 768), (151296, 768, 1), 0); del buf87  # reuse
        # Source Nodes: [mul_9, x_80, x_81], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf101, arg48_1, buf108, arg154_1, arg52_1, arg53_1, buf112, 1576, 768, grid=grid(1576), stream=stream0)
        del arg52_1
        del arg53_1
        buf113 = buf92; del buf92  # reuse
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg54_1, arg209_1, arg55_1, buf113, 2304, grid=grid(2304), stream=stream0)
        del arg209_1
        del arg54_1
        del arg55_1
        buf114 = buf93; del buf93  # reuse
        # Source Nodes: [cat_19, qkv_10], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf113, reinterpret_tensor(buf112, (1576, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf114)
        del arg56_1
        buf115 = buf94; del buf94  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg210_1, arg57_1, buf115, 472800, grid=grid(472800), stream=stream0)
        del arg210_1
        del arg57_1
        # Source Nodes: [x_82], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf116 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf115, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf117 = buf116[0]
        del buf116
        buf121 = reinterpret_tensor(buf112, (1576, 768), (768, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (1576, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), out=buf121)
        del arg155_1
        buf122 = reinterpret_tensor(buf121, (8, 197, 768), (151296, 768, 1), 0); del buf121  # reuse
        buf126 = reinterpret_tensor(buf117, (8, 197, 768), (151296, 768, 1), 0); del buf117  # reuse
        # Source Nodes: [mul_10, mul_9, x_80, x_86, x_87], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf122, buf101, arg48_1, buf108, arg154_1, arg51_1, arg156_1, arg59_1, arg60_1, buf126, 1576, 768, grid=grid(1576), stream=stream0)
        del arg154_1
        del arg156_1
        del arg48_1
        del arg51_1
        del arg59_1
        del arg60_1
        del buf101
        buf127 = reinterpret_tensor(buf107, (1576, 3072), (3072, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (1576, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 3072), (1, 768), 0), out=buf127)
        del arg157_1
        buf128 = reinterpret_tensor(buf127, (8, 197, 3072), (605184, 3072, 1), 0); del buf127  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf128, arg158_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg158_1
        buf129 = reinterpret_tensor(buf126, (1576, 768), (768, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg159_1, (3072, 768), (1, 3072), 0), out=buf129)
        del arg159_1
        buf133 = reinterpret_tensor(buf108, (8, 197, 768), (151296, 768, 1), 0); del buf108  # reuse
        # Source Nodes: [mul_11, x_95, x_96], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf122, arg58_1, buf129, arg160_1, arg62_1, arg63_1, buf133, 1576, 768, grid=grid(1576), stream=stream0)
        del arg62_1
        del arg63_1
        buf134 = buf113; del buf113  # reuse
        # Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg64_1, arg211_1, arg65_1, buf134, 2304, grid=grid(2304), stream=stream0)
        del arg211_1
        del arg64_1
        del arg65_1
        buf135 = buf114; del buf114  # reuse
        # Source Nodes: [cat_18, qkv_12], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf134, reinterpret_tensor(buf133, (1576, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf135)
        del arg66_1
        buf136 = buf115; del buf115  # reuse
        # Source Nodes: [x_97], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg212_1, arg67_1, buf136, 472800, grid=grid(472800), stream=stream0)
        del arg212_1
        del arg67_1
        # Source Nodes: [x_97], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf137 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf136, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf138 = buf137[0]
        del buf137
        buf142 = reinterpret_tensor(buf133, (1576, 768), (768, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (1576, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 768), (1, 768), 0), out=buf142)
        del arg161_1
        buf143 = reinterpret_tensor(buf142, (8, 197, 768), (151296, 768, 1), 0); del buf142  # reuse
        buf147 = reinterpret_tensor(buf138, (8, 197, 768), (151296, 768, 1), 0); del buf138  # reuse
        # Source Nodes: [mul_11, mul_12, x_101, x_102, x_95], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf143, buf122, arg58_1, buf129, arg160_1, arg61_1, arg162_1, arg69_1, arg70_1, buf147, 1576, 768, grid=grid(1576), stream=stream0)
        del arg160_1
        del arg162_1
        del arg58_1
        del arg61_1
        del arg69_1
        del arg70_1
        del buf122
        buf148 = reinterpret_tensor(buf128, (1576, 3072), (3072, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1576, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 3072), (1, 768), 0), out=buf148)
        del arg163_1
        buf149 = reinterpret_tensor(buf148, (8, 197, 3072), (605184, 3072, 1), 0); del buf148  # reuse
        # Source Nodes: [x_104], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf149, arg164_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg164_1
        buf150 = reinterpret_tensor(buf147, (1576, 768), (768, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg165_1, (3072, 768), (1, 3072), 0), out=buf150)
        del arg165_1
        buf154 = reinterpret_tensor(buf129, (8, 197, 768), (151296, 768, 1), 0); del buf129  # reuse
        # Source Nodes: [mul_13, x_110, x_111], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf143, arg68_1, buf150, arg166_1, arg72_1, arg73_1, buf154, 1576, 768, grid=grid(1576), stream=stream0)
        del arg72_1
        del arg73_1
        buf155 = buf134; del buf134  # reuse
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg74_1, arg213_1, arg75_1, buf155, 2304, grid=grid(2304), stream=stream0)
        del arg213_1
        del arg74_1
        del arg75_1
        buf156 = buf135; del buf135  # reuse
        # Source Nodes: [cat_17, qkv_14], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf155, reinterpret_tensor(buf154, (1576, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf156)
        del arg76_1
        buf157 = buf136; del buf136  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg214_1, arg77_1, buf157, 472800, grid=grid(472800), stream=stream0)
        del arg214_1
        del arg77_1
        # Source Nodes: [x_112], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf158 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf157, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf159 = buf158[0]
        del buf158
        buf163 = reinterpret_tensor(buf154, (1576, 768), (768, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (1576, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 768), (1, 768), 0), out=buf163)
        del arg167_1
        buf164 = reinterpret_tensor(buf163, (8, 197, 768), (151296, 768, 1), 0); del buf163  # reuse
        buf168 = reinterpret_tensor(buf159, (8, 197, 768), (151296, 768, 1), 0); del buf159  # reuse
        # Source Nodes: [mul_13, mul_14, x_110, x_116, x_117], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf164, buf143, arg68_1, buf150, arg166_1, arg71_1, arg168_1, arg79_1, arg80_1, buf168, 1576, 768, grid=grid(1576), stream=stream0)
        del arg166_1
        del arg168_1
        del arg68_1
        del arg71_1
        del arg79_1
        del arg80_1
        del buf143
        buf169 = reinterpret_tensor(buf149, (1576, 3072), (3072, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (1576, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 3072), (1, 768), 0), out=buf169)
        del arg169_1
        buf170 = reinterpret_tensor(buf169, (8, 197, 3072), (605184, 3072, 1), 0); del buf169  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf170, arg170_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg170_1
        buf171 = reinterpret_tensor(buf168, (1576, 768), (768, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg171_1, (3072, 768), (1, 3072), 0), out=buf171)
        del arg171_1
        buf175 = reinterpret_tensor(buf150, (8, 197, 768), (151296, 768, 1), 0); del buf150  # reuse
        # Source Nodes: [mul_15, x_125, x_126], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf164, arg78_1, buf171, arg172_1, arg82_1, arg83_1, buf175, 1576, 768, grid=grid(1576), stream=stream0)
        del arg82_1
        del arg83_1
        buf176 = buf155; del buf155  # reuse
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg84_1, arg215_1, arg85_1, buf176, 2304, grid=grid(2304), stream=stream0)
        del arg215_1
        del arg84_1
        del arg85_1
        buf177 = buf156; del buf156  # reuse
        # Source Nodes: [cat_16, qkv_16], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf176, reinterpret_tensor(buf175, (1576, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf177)
        del arg86_1
        buf178 = buf157; del buf157  # reuse
        # Source Nodes: [x_127], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg216_1, arg87_1, buf178, 472800, grid=grid(472800), stream=stream0)
        del arg216_1
        del arg87_1
        # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf179 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf178, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf180 = buf179[0]
        del buf179
        buf184 = reinterpret_tensor(buf175, (1576, 768), (768, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (1576, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), out=buf184)
        del arg173_1
        buf185 = reinterpret_tensor(buf184, (8, 197, 768), (151296, 768, 1), 0); del buf184  # reuse
        buf189 = reinterpret_tensor(buf180, (8, 197, 768), (151296, 768, 1), 0); del buf180  # reuse
        # Source Nodes: [mul_15, mul_16, x_125, x_131, x_132], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf185, buf164, arg78_1, buf171, arg172_1, arg81_1, arg174_1, arg89_1, arg90_1, buf189, 1576, 768, grid=grid(1576), stream=stream0)
        del arg172_1
        del arg174_1
        del arg78_1
        del arg81_1
        del arg89_1
        del arg90_1
        del buf164
        buf190 = reinterpret_tensor(buf170, (1576, 3072), (3072, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (1576, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 3072), (1, 768), 0), out=buf190)
        del arg175_1
        buf191 = reinterpret_tensor(buf190, (8, 197, 3072), (605184, 3072, 1), 0); del buf190  # reuse
        # Source Nodes: [x_134], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf191, arg176_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg176_1
        buf192 = reinterpret_tensor(buf189, (1576, 768), (768, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg177_1, (3072, 768), (1, 3072), 0), out=buf192)
        del arg177_1
        buf196 = reinterpret_tensor(buf171, (8, 197, 768), (151296, 768, 1), 0); del buf171  # reuse
        # Source Nodes: [mul_17, x_140, x_141], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf185, arg88_1, buf192, arg178_1, arg92_1, arg93_1, buf196, 1576, 768, grid=grid(1576), stream=stream0)
        del arg92_1
        del arg93_1
        buf197 = buf176; del buf176  # reuse
        # Source Nodes: [cat_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg94_1, arg217_1, arg95_1, buf197, 2304, grid=grid(2304), stream=stream0)
        del arg217_1
        del arg94_1
        del arg95_1
        buf198 = buf177; del buf177  # reuse
        # Source Nodes: [cat_15, qkv_18], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf197, reinterpret_tensor(buf196, (1576, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf198)
        del arg96_1
        buf199 = buf178; del buf178  # reuse
        # Source Nodes: [x_142], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg218_1, arg97_1, buf199, 472800, grid=grid(472800), stream=stream0)
        del arg218_1
        del arg97_1
        # Source Nodes: [x_142], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf200 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf199, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf201 = buf200[0]
        del buf200
        buf205 = reinterpret_tensor(buf196, (1576, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (1576, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 768), (1, 768), 0), out=buf205)
        del arg179_1
        buf206 = reinterpret_tensor(buf205, (8, 197, 768), (151296, 768, 1), 0); del buf205  # reuse
        buf210 = reinterpret_tensor(buf201, (8, 197, 768), (151296, 768, 1), 0); del buf201  # reuse
        # Source Nodes: [mul_17, mul_18, x_140, x_146, x_147], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf206, buf185, arg88_1, buf192, arg178_1, arg91_1, arg180_1, arg99_1, arg100_1, buf210, 1576, 768, grid=grid(1576), stream=stream0)
        del arg100_1
        del arg178_1
        del arg180_1
        del arg88_1
        del arg91_1
        del arg99_1
        del buf185
        buf211 = reinterpret_tensor(buf191, (1576, 3072), (3072, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (1576, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 3072), (1, 768), 0), out=buf211)
        del arg181_1
        buf212 = reinterpret_tensor(buf211, (8, 197, 3072), (605184, 3072, 1), 0); del buf211  # reuse
        # Source Nodes: [x_149], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf212, arg182_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg182_1
        buf213 = reinterpret_tensor(buf210, (1576, 768), (768, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg183_1, (3072, 768), (1, 3072), 0), out=buf213)
        del arg183_1
        buf217 = reinterpret_tensor(buf192, (8, 197, 768), (151296, 768, 1), 0); del buf192  # reuse
        # Source Nodes: [mul_19, x_155, x_156], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf206, arg98_1, buf213, arg184_1, arg102_1, arg103_1, buf217, 1576, 768, grid=grid(1576), stream=stream0)
        del arg102_1
        del arg103_1
        buf218 = buf197; del buf197  # reuse
        # Source Nodes: [cat_14], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg104_1, arg219_1, arg105_1, buf218, 2304, grid=grid(2304), stream=stream0)
        del arg104_1
        del arg105_1
        del arg219_1
        buf219 = buf198; del buf198  # reuse
        # Source Nodes: [cat_14, qkv_20], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf218, reinterpret_tensor(buf217, (1576, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf219)
        del arg106_1
        buf220 = buf199; del buf199  # reuse
        # Source Nodes: [x_157], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg220_1, arg107_1, buf220, 472800, grid=grid(472800), stream=stream0)
        del arg107_1
        del arg220_1
        # Source Nodes: [x_157], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf221 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf220, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        buf222 = buf221[0]
        del buf221
        buf226 = reinterpret_tensor(buf217, (1576, 768), (768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf222, (1576, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), out=buf226)
        del arg185_1
        buf227 = reinterpret_tensor(buf226, (8, 197, 768), (151296, 768, 1), 0); del buf226  # reuse
        buf231 = reinterpret_tensor(buf222, (8, 197, 768), (151296, 768, 1), 0); del buf222  # reuse
        # Source Nodes: [mul_19, mul_20, x_155, x_161, x_162], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf227, buf206, arg98_1, buf213, arg184_1, arg101_1, arg186_1, arg109_1, arg110_1, buf231, 1576, 768, grid=grid(1576), stream=stream0)
        del arg101_1
        del arg109_1
        del arg110_1
        del arg184_1
        del arg186_1
        del arg98_1
        del buf206
        buf232 = reinterpret_tensor(buf212, (1576, 3072), (3072, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (1576, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 3072), (1, 768), 0), out=buf232)
        del arg187_1
        buf233 = reinterpret_tensor(buf232, (8, 197, 3072), (605184, 3072, 1), 0); del buf232  # reuse
        # Source Nodes: [x_164], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf233, arg188_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg188_1
        buf234 = reinterpret_tensor(buf231, (1576, 768), (768, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg189_1, (3072, 768), (1, 3072), 0), out=buf234)
        del arg189_1
        buf238 = reinterpret_tensor(buf213, (8, 197, 768), (151296, 768, 1), 0); del buf213  # reuse
        # Source Nodes: [mul_21, x_170, x_171], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_7.run(buf227, arg108_1, buf234, arg190_1, arg112_1, arg113_1, buf238, 1576, 768, grid=grid(1576), stream=stream0)
        del arg112_1
        del arg113_1
        buf239 = buf218; del buf218  # reuse
        # Source Nodes: [cat_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg114_1, arg221_1, arg115_1, buf239, 2304, grid=grid(2304), stream=stream0)
        del arg114_1
        del arg115_1
        del arg221_1
        buf240 = buf219; del buf219  # reuse
        # Source Nodes: [cat_13, qkv_22], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf239, reinterpret_tensor(buf238, (1576, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf240)
        del arg116_1
        del buf239
        buf241 = buf220; del buf220  # reuse
        # Source Nodes: [x_172], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_4.run(arg222_1, arg117_1, buf241, 472800, grid=grid(472800), stream=stream0)
        del arg117_1
        del arg222_1
        # Source Nodes: [x_172], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf242 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf241, (8, 12, 197, 197), (0, 39400, 200, 1), 0), False)
        del buf240
        del buf241
        buf243 = buf242[0]
        del buf242
        buf247 = reinterpret_tensor(buf238, (1576, 768), (768, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (1576, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 768), (1, 768), 0), out=buf247)
        del arg191_1
        buf248 = reinterpret_tensor(buf247, (8, 197, 768), (151296, 768, 1), 0); del buf247  # reuse
        buf252 = reinterpret_tensor(buf243, (8, 197, 768), (151296, 768, 1), 0); del buf243  # reuse
        # Source Nodes: [mul_21, mul_22, x_170, x_176, x_177], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_8.run(buf248, buf227, arg108_1, buf234, arg190_1, arg111_1, arg192_1, arg119_1, arg120_1, buf252, 1576, 768, grid=grid(1576), stream=stream0)
        del arg108_1
        del arg111_1
        del arg119_1
        del arg120_1
        del arg190_1
        del arg192_1
        del buf227
        del buf234
        buf253 = reinterpret_tensor(buf233, (1576, 3072), (3072, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (1576, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 3072), (1, 768), 0), out=buf253)
        del arg193_1
        buf254 = reinterpret_tensor(buf253, (8, 197, 3072), (605184, 3072, 1), 0); del buf253  # reuse
        # Source Nodes: [x_179], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf254, arg194_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg194_1
        buf255 = reinterpret_tensor(buf252, (1576, 768), (768, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf254, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg195_1, (3072, 768), (1, 3072), 0), out=buf255)
        del arg195_1
        del buf254
        buf256 = empty_strided((8, 768, 2), (1536, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.mean]
        triton_red_fused_mean_9.run(buf248, arg118_1, buf255, arg196_1, buf256, 12288, 98, grid=grid(12288), stream=stream0)
        del arg118_1
        del arg196_1
        del buf248
        del buf255
        buf257 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.mean]
        triton_per_fused_mean_10.run(buf256, buf257, 6144, 2, grid=grid(6144), stream=stream0)
        del buf256
        buf261 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188, x_190], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_11.run(buf257, arg121_1, arg122_1, buf261, 8, 768, grid=grid(8), stream=stream0)
        del arg121_1
        del arg122_1
        del buf257
        buf262 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188, x_190, x_192], Original ATen: [aten.addmm, aten.mean, aten.native_layer_norm]
        extern_kernels.addmm(arg198_1, buf261, reinterpret_tensor(arg197_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf262)
        del arg197_1
        del arg198_1
        return (buf262, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg205_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg209_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg211_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg219_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg221_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    arg223_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('beit_base_patch16_224', benchmark_compiled_module)
