
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


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpwmybtchvvrlxqksd3nbkac24mhwvpxpjcsyha4zfxjixh6qg5.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# getattr_l__mod___blocks___0___norm1 => var_mean
# x_5 => add
triton_per_fused_add_cat_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_1', 'mutated_arg_names': []}
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdiexpclj4n27zuv4jjc4fktgfzj6hywg3lh7k3x5aguzxz2bqdk.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# getattr_l__mod___blocks___0___norm1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# x_5 => add
triton_poi_fused_add_cat_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_layer_norm_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x5), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czoyx3lv33invlj3yjdvoadjjc2cp2v7wasbc53depqsq4e4q6ks.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm2, x_13, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# getattr_l__mod___blocks___0___norm2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# x_13 => add_3
# x_5 => add
triton_per_fused_add_cat_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0']}
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
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crcjewaotnvl2to6yigzs5mnif5jdsa4l7qleu3yxsn5d6cksdk7.py
# Source Nodes: [x_15], Original ATen: [aten.gelu]
# x_15 => add_6, erf, mul_4, mul_5, mul_6
triton_poi_fused_gelu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcvd6sg7i37krvptyzxgk2wbrcrewsonkwsr7gj2zktogswqiu5.py
# Source Nodes: [getattr_l__mod___blocks___1___norm1, x_20], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm1 => add_8, add_9, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
# x_20 => add_7
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cruzr5reg3bamk2mkryb3xat6dy3cojb5rvhlbhlhzjcsq5jtkeh.py
# Source Nodes: [getattr_l__mod___blocks___1___norm2, x_20, x_25], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm2 => add_11, add_12, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
# x_20 => add_7
# x_25 => add_10
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cah5de5yk2kb7oxxp6nwefvxvwc2nfyh6n7gyjsmb44qkaa5w67w.py
# Source Nodes: [x_153, x_155], Original ATen: [aten.add, aten.native_layer_norm]
# x_153 => add_84
# x_155 => var_mean_24
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6cfmze2vzo6hkltgv5x7csieiv4zpwkg2bozlxwcdxqe3wcytce.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (151296*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (151296*x1)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (197*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (197*x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 197, 768), (151296, 768, 1))
    assert_size_stride(arg1_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg2_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (2304, 768), (768, 1))
    assert_size_stride(arg7_1, (2304, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (3072, 768), (768, 1))
    assert_size_stride(arg13_1, (3072, ), (1, ))
    assert_size_stride(arg14_1, (768, 3072), (3072, 1))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (2304, 768), (768, 1))
    assert_size_stride(arg19_1, (2304, ), (1, ))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (3072, 768), (768, 1))
    assert_size_stride(arg25_1, (3072, ), (1, ))
    assert_size_stride(arg26_1, (768, 3072), (3072, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (2304, 768), (768, 1))
    assert_size_stride(arg31_1, (2304, ), (1, ))
    assert_size_stride(arg32_1, (768, 768), (768, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (3072, 768), (768, 1))
    assert_size_stride(arg37_1, (3072, ), (1, ))
    assert_size_stride(arg38_1, (768, 3072), (3072, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (2304, 768), (768, 1))
    assert_size_stride(arg43_1, (2304, ), (1, ))
    assert_size_stride(arg44_1, (768, 768), (768, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (3072, 768), (768, 1))
    assert_size_stride(arg49_1, (3072, ), (1, ))
    assert_size_stride(arg50_1, (768, 3072), (3072, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (2304, 768), (768, 1))
    assert_size_stride(arg55_1, (2304, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (3072, 768), (768, 1))
    assert_size_stride(arg61_1, (3072, ), (1, ))
    assert_size_stride(arg62_1, (768, 3072), (3072, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (2304, 768), (768, 1))
    assert_size_stride(arg67_1, (2304, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (3072, 768), (768, 1))
    assert_size_stride(arg73_1, (3072, ), (1, ))
    assert_size_stride(arg74_1, (768, 3072), (3072, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (2304, 768), (768, 1))
    assert_size_stride(arg79_1, (2304, ), (1, ))
    assert_size_stride(arg80_1, (768, 768), (768, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (3072, 768), (768, 1))
    assert_size_stride(arg85_1, (3072, ), (1, ))
    assert_size_stride(arg86_1, (768, 3072), (3072, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (2304, 768), (768, 1))
    assert_size_stride(arg91_1, (2304, ), (1, ))
    assert_size_stride(arg92_1, (768, 768), (768, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (3072, 768), (768, 1))
    assert_size_stride(arg97_1, (3072, ), (1, ))
    assert_size_stride(arg98_1, (768, 3072), (3072, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (2304, 768), (768, 1))
    assert_size_stride(arg103_1, (2304, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (3072, 768), (768, 1))
    assert_size_stride(arg109_1, (3072, ), (1, ))
    assert_size_stride(arg110_1, (768, 3072), (3072, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (2304, 768), (768, 1))
    assert_size_stride(arg115_1, (2304, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (3072, 768), (768, 1))
    assert_size_stride(arg121_1, (3072, ), (1, ))
    assert_size_stride(arg122_1, (768, 3072), (3072, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (2304, 768), (768, 1))
    assert_size_stride(arg127_1, (2304, ), (1, ))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (3072, 768), (768, 1))
    assert_size_stride(arg133_1, (3072, ), (1, ))
    assert_size_stride(arg134_1, (768, 3072), (3072, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (2304, 768), (768, 1))
    assert_size_stride(arg139_1, (2304, ), (1, ))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (3072, 768), (768, 1))
    assert_size_stride(arg145_1, (3072, ), (1, ))
    assert_size_stride(arg146_1, (768, 3072), (3072, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (1000, 768), (768, 1))
    assert_size_stride(arg151_1, (1000, ), (1, ))
    assert_size_stride(arg152_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg152_1, arg2_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg152_1
        del arg2_1
        buf1 = empty_strided((8, 197, 1, 6), (1182, 6, 9456, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 197, 1, 6), (1182, 6, 9456, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 197, 1, 6), (1182, 6, 9456, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_cat_native_layer_norm_0.run(arg1_1, buf0, arg3_1, arg0_1, buf1, buf2, buf3, 9456, 128, grid=grid(9456), stream=stream0)
        buf4 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1576, 6, grid=grid(1576), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_poi_fused_add_cat_native_layer_norm_2.run(arg1_1, buf0, arg3_1, arg0_1, buf4, buf5, arg4_1, arg5_1, buf7, 1210368, grid=grid(1210368), stream=stream0)
        del arg4_1
        del arg5_1
        buf8 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg7_1, reinterpret_tensor(buf7, (1576, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf8)
        del arg6_1
        del arg7_1
        # Source Nodes: [x_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf9 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf10 = buf9[0]
        del buf9
        buf14 = reinterpret_tensor(buf7, (1576, 768), (768, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf10, (1576, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), out=buf14)
        del arg8_1
        buf15 = reinterpret_tensor(buf14, (8, 197, 768), (151296, 768, 1), 0); del buf14  # reuse
        buf19 = reinterpret_tensor(buf10, (8, 197, 768), (151296, 768, 1), 0); del buf10  # reuse
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm2, x_13, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_3.run(buf15, arg1_1, buf0, arg3_1, arg0_1, arg9_1, arg10_1, arg11_1, buf19, 1576, 768, grid=grid(1576), stream=stream0)
        del arg0_1
        del arg10_1
        del arg11_1
        del arg1_1
        del arg3_1
        del arg9_1
        del buf0
        buf20 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (1576, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 3072), (1, 768), 0), out=buf20)
        del arg12_1
        buf21 = reinterpret_tensor(buf20, (8, 197, 3072), (605184, 3072, 1), 0); del buf20  # reuse
        # Source Nodes: [x_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf21, arg13_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg13_1
        buf22 = reinterpret_tensor(buf19, (1576, 768), (768, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg14_1, (3072, 768), (1, 3072), 0), out=buf22)
        del arg14_1
        buf26 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm1, x_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf15, buf22, arg15_1, arg16_1, arg17_1, buf26, 1576, 768, grid=grid(1576), stream=stream0)
        del arg16_1
        del arg17_1
        buf27 = buf8; del buf8  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, reinterpret_tensor(buf26, (1576, 768), (768, 1), 0), reinterpret_tensor(arg18_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf27)
        del arg18_1
        del arg19_1
        # Source Nodes: [x_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf28 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf27, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf27, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf27, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf29 = buf28[0]
        del buf28
        buf33 = reinterpret_tensor(buf26, (1576, 768), (768, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (1576, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 768), (1, 768), 0), out=buf33)
        del arg20_1
        buf34 = reinterpret_tensor(buf33, (8, 197, 768), (151296, 768, 1), 0); del buf33  # reuse
        buf38 = reinterpret_tensor(buf29, (8, 197, 768), (151296, 768, 1), 0); del buf29  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_20, x_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf34, buf15, buf22, arg15_1, arg21_1, arg22_1, arg23_1, buf38, 1576, 768, grid=grid(1576), stream=stream0)
        del arg15_1
        del arg21_1
        del arg22_1
        del arg23_1
        del buf15
        buf39 = reinterpret_tensor(buf21, (1576, 3072), (3072, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (1576, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 3072), (1, 768), 0), out=buf39)
        del arg24_1
        buf40 = reinterpret_tensor(buf39, (8, 197, 3072), (605184, 3072, 1), 0); del buf39  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf40, arg25_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg25_1
        buf41 = reinterpret_tensor(buf38, (1576, 768), (768, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg26_1, (3072, 768), (1, 3072), 0), out=buf41)
        del arg26_1
        buf45 = reinterpret_tensor(buf22, (8, 197, 768), (151296, 768, 1), 0); del buf22  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm1, x_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf34, buf41, arg27_1, arg28_1, arg29_1, buf45, 1576, 768, grid=grid(1576), stream=stream0)
        del arg28_1
        del arg29_1
        buf46 = buf27; del buf27  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg31_1, reinterpret_tensor(buf45, (1576, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf46)
        del arg30_1
        del arg31_1
        # Source Nodes: [x_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf47 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf46, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf46, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf46, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf48 = buf47[0]
        del buf47
        buf52 = reinterpret_tensor(buf45, (1576, 768), (768, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (1576, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 768), (1, 768), 0), out=buf52)
        del arg32_1
        buf53 = reinterpret_tensor(buf52, (8, 197, 768), (151296, 768, 1), 0); del buf52  # reuse
        buf57 = reinterpret_tensor(buf48, (8, 197, 768), (151296, 768, 1), 0); del buf48  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_32, x_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf53, buf34, buf41, arg27_1, arg33_1, arg34_1, arg35_1, buf57, 1576, 768, grid=grid(1576), stream=stream0)
        del arg27_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf34
        buf58 = reinterpret_tensor(buf40, (1576, 3072), (3072, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (1576, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 3072), (1, 768), 0), out=buf58)
        del arg36_1
        buf59 = reinterpret_tensor(buf58, (8, 197, 3072), (605184, 3072, 1), 0); del buf58  # reuse
        # Source Nodes: [x_39], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf59, arg37_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg37_1
        buf60 = reinterpret_tensor(buf57, (1576, 768), (768, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg38_1, (3072, 768), (1, 3072), 0), out=buf60)
        del arg38_1
        buf64 = reinterpret_tensor(buf41, (8, 197, 768), (151296, 768, 1), 0); del buf41  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm1, x_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf53, buf60, arg39_1, arg40_1, arg41_1, buf64, 1576, 768, grid=grid(1576), stream=stream0)
        del arg40_1
        del arg41_1
        buf65 = buf46; del buf46  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg43_1, reinterpret_tensor(buf64, (1576, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf65)
        del arg42_1
        del arg43_1
        # Source Nodes: [x_45], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf66 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf65, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf65, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf65, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf67 = buf66[0]
        del buf66
        buf71 = reinterpret_tensor(buf64, (1576, 768), (768, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (1576, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0), out=buf71)
        del arg44_1
        buf72 = reinterpret_tensor(buf71, (8, 197, 768), (151296, 768, 1), 0); del buf71  # reuse
        buf76 = reinterpret_tensor(buf67, (8, 197, 768), (151296, 768, 1), 0); del buf67  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_44, x_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf72, buf53, buf60, arg39_1, arg45_1, arg46_1, arg47_1, buf76, 1576, 768, grid=grid(1576), stream=stream0)
        del arg39_1
        del arg45_1
        del arg46_1
        del arg47_1
        del buf53
        buf77 = reinterpret_tensor(buf59, (1576, 3072), (3072, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (1576, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 3072), (1, 768), 0), out=buf77)
        del arg48_1
        buf78 = reinterpret_tensor(buf77, (8, 197, 3072), (605184, 3072, 1), 0); del buf77  # reuse
        # Source Nodes: [x_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf78, arg49_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg49_1
        buf79 = reinterpret_tensor(buf76, (1576, 768), (768, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg50_1, (3072, 768), (1, 3072), 0), out=buf79)
        del arg50_1
        buf83 = reinterpret_tensor(buf60, (8, 197, 768), (151296, 768, 1), 0); del buf60  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf72, buf79, arg51_1, arg52_1, arg53_1, buf83, 1576, 768, grid=grid(1576), stream=stream0)
        del arg52_1
        del arg53_1
        buf84 = buf65; del buf65  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg55_1, reinterpret_tensor(buf83, (1576, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf84)
        del arg54_1
        del arg55_1
        # Source Nodes: [x_57], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf85 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf84, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf84, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf84, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf86 = buf85[0]
        del buf85
        buf90 = reinterpret_tensor(buf83, (1576, 768), (768, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf86, (1576, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), out=buf90)
        del arg56_1
        buf91 = reinterpret_tensor(buf90, (8, 197, 768), (151296, 768, 1), 0); del buf90  # reuse
        buf95 = reinterpret_tensor(buf86, (8, 197, 768), (151296, 768, 1), 0); del buf86  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_56, x_61], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf91, buf72, buf79, arg51_1, arg57_1, arg58_1, arg59_1, buf95, 1576, 768, grid=grid(1576), stream=stream0)
        del arg51_1
        del arg57_1
        del arg58_1
        del arg59_1
        del buf72
        buf96 = reinterpret_tensor(buf78, (1576, 3072), (3072, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (1576, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 3072), (1, 768), 0), out=buf96)
        del arg60_1
        buf97 = reinterpret_tensor(buf96, (8, 197, 3072), (605184, 3072, 1), 0); del buf96  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf97, arg61_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg61_1
        buf98 = reinterpret_tensor(buf95, (1576, 768), (768, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg62_1, (3072, 768), (1, 3072), 0), out=buf98)
        del arg62_1
        buf102 = reinterpret_tensor(buf79, (8, 197, 768), (151296, 768, 1), 0); del buf79  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm1, x_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf91, buf98, arg63_1, arg64_1, arg65_1, buf102, 1576, 768, grid=grid(1576), stream=stream0)
        del arg64_1
        del arg65_1
        buf103 = buf84; del buf84  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg67_1, reinterpret_tensor(buf102, (1576, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf103)
        del arg66_1
        del arg67_1
        # Source Nodes: [x_69], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf104 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf103, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf103, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf103, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf105 = buf104[0]
        del buf104
        buf109 = reinterpret_tensor(buf102, (1576, 768), (768, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (1576, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), out=buf109)
        del arg68_1
        buf110 = reinterpret_tensor(buf109, (8, 197, 768), (151296, 768, 1), 0); del buf109  # reuse
        buf114 = reinterpret_tensor(buf105, (8, 197, 768), (151296, 768, 1), 0); del buf105  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_68, x_73], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf110, buf91, buf98, arg63_1, arg69_1, arg70_1, arg71_1, buf114, 1576, 768, grid=grid(1576), stream=stream0)
        del arg63_1
        del arg69_1
        del arg70_1
        del arg71_1
        del buf91
        buf115 = reinterpret_tensor(buf97, (1576, 3072), (3072, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (1576, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 3072), (1, 768), 0), out=buf115)
        del arg72_1
        buf116 = reinterpret_tensor(buf115, (8, 197, 3072), (605184, 3072, 1), 0); del buf115  # reuse
        # Source Nodes: [x_75], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf116, arg73_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg73_1
        buf117 = reinterpret_tensor(buf114, (1576, 768), (768, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg74_1, (3072, 768), (1, 3072), 0), out=buf117)
        del arg74_1
        buf121 = reinterpret_tensor(buf98, (8, 197, 768), (151296, 768, 1), 0); del buf98  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm1, x_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf110, buf117, arg75_1, arg76_1, arg77_1, buf121, 1576, 768, grid=grid(1576), stream=stream0)
        del arg76_1
        del arg77_1
        buf122 = buf103; del buf103  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg79_1, reinterpret_tensor(buf121, (1576, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf122)
        del arg78_1
        del arg79_1
        # Source Nodes: [x_81], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf123 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf122, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf122, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf122, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf124 = buf123[0]
        del buf123
        buf128 = reinterpret_tensor(buf121, (1576, 768), (768, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1576, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 768), (1, 768), 0), out=buf128)
        del arg80_1
        buf129 = reinterpret_tensor(buf128, (8, 197, 768), (151296, 768, 1), 0); del buf128  # reuse
        buf133 = reinterpret_tensor(buf124, (8, 197, 768), (151296, 768, 1), 0); del buf124  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_80, x_85], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf129, buf110, buf117, arg75_1, arg81_1, arg82_1, arg83_1, buf133, 1576, 768, grid=grid(1576), stream=stream0)
        del arg75_1
        del arg81_1
        del arg82_1
        del arg83_1
        del buf110
        buf134 = reinterpret_tensor(buf116, (1576, 3072), (3072, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf133, (1576, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 3072), (1, 768), 0), out=buf134)
        del arg84_1
        buf135 = reinterpret_tensor(buf134, (8, 197, 3072), (605184, 3072, 1), 0); del buf134  # reuse
        # Source Nodes: [x_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf135, arg85_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg85_1
        buf136 = reinterpret_tensor(buf133, (1576, 768), (768, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg86_1, (3072, 768), (1, 3072), 0), out=buf136)
        del arg86_1
        buf140 = reinterpret_tensor(buf117, (8, 197, 768), (151296, 768, 1), 0); del buf117  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm1, x_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf129, buf136, arg87_1, arg88_1, arg89_1, buf140, 1576, 768, grid=grid(1576), stream=stream0)
        del arg88_1
        del arg89_1
        buf141 = buf122; del buf122  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg91_1, reinterpret_tensor(buf140, (1576, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf141)
        del arg90_1
        del arg91_1
        # Source Nodes: [x_93], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf142 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf141, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf141, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf141, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf143 = buf142[0]
        del buf142
        buf147 = reinterpret_tensor(buf140, (1576, 768), (768, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (1576, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), out=buf147)
        del arg92_1
        buf148 = reinterpret_tensor(buf147, (8, 197, 768), (151296, 768, 1), 0); del buf147  # reuse
        buf152 = reinterpret_tensor(buf143, (8, 197, 768), (151296, 768, 1), 0); del buf143  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_92, x_97], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf148, buf129, buf136, arg87_1, arg93_1, arg94_1, arg95_1, buf152, 1576, 768, grid=grid(1576), stream=stream0)
        del arg87_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf129
        buf153 = reinterpret_tensor(buf135, (1576, 3072), (3072, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (1576, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 3072), (1, 768), 0), out=buf153)
        del arg96_1
        buf154 = reinterpret_tensor(buf153, (8, 197, 3072), (605184, 3072, 1), 0); del buf153  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf154, arg97_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg97_1
        buf155 = reinterpret_tensor(buf152, (1576, 768), (768, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg98_1, (3072, 768), (1, 3072), 0), out=buf155)
        del arg98_1
        buf159 = reinterpret_tensor(buf136, (8, 197, 768), (151296, 768, 1), 0); del buf136  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm1, x_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf148, buf155, arg99_1, arg100_1, arg101_1, buf159, 1576, 768, grid=grid(1576), stream=stream0)
        del arg100_1
        del arg101_1
        buf160 = buf141; del buf141  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg103_1, reinterpret_tensor(buf159, (1576, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf160)
        del arg102_1
        del arg103_1
        # Source Nodes: [x_105], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf161 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf160, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf160, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf160, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf162 = buf161[0]
        del buf161
        buf166 = reinterpret_tensor(buf159, (1576, 768), (768, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (1576, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), out=buf166)
        del arg104_1
        buf167 = reinterpret_tensor(buf166, (8, 197, 768), (151296, 768, 1), 0); del buf166  # reuse
        buf171 = reinterpret_tensor(buf162, (8, 197, 768), (151296, 768, 1), 0); del buf162  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_104, x_109], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf167, buf148, buf155, arg99_1, arg105_1, arg106_1, arg107_1, buf171, 1576, 768, grid=grid(1576), stream=stream0)
        del arg105_1
        del arg106_1
        del arg107_1
        del arg99_1
        del buf148
        buf172 = reinterpret_tensor(buf154, (1576, 3072), (3072, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1576, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 3072), (1, 768), 0), out=buf172)
        del arg108_1
        buf173 = reinterpret_tensor(buf172, (8, 197, 3072), (605184, 3072, 1), 0); del buf172  # reuse
        # Source Nodes: [x_111], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf173, arg109_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg109_1
        buf174 = reinterpret_tensor(buf171, (1576, 768), (768, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg110_1, (3072, 768), (1, 3072), 0), out=buf174)
        del arg110_1
        buf178 = reinterpret_tensor(buf155, (8, 197, 768), (151296, 768, 1), 0); del buf155  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm1, x_116], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf167, buf174, arg111_1, arg112_1, arg113_1, buf178, 1576, 768, grid=grid(1576), stream=stream0)
        del arg112_1
        del arg113_1
        buf179 = buf160; del buf160  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg115_1, reinterpret_tensor(buf178, (1576, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf179)
        del arg114_1
        del arg115_1
        # Source Nodes: [x_117], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf180 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf179, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf179, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf179, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf181 = buf180[0]
        del buf180
        buf185 = reinterpret_tensor(buf178, (1576, 768), (768, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf181, (1576, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), out=buf185)
        del arg116_1
        buf186 = reinterpret_tensor(buf185, (8, 197, 768), (151296, 768, 1), 0); del buf185  # reuse
        buf190 = reinterpret_tensor(buf181, (8, 197, 768), (151296, 768, 1), 0); del buf181  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_116, x_121], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf186, buf167, buf174, arg111_1, arg117_1, arg118_1, arg119_1, buf190, 1576, 768, grid=grid(1576), stream=stream0)
        del arg111_1
        del arg117_1
        del arg118_1
        del arg119_1
        del buf167
        buf191 = reinterpret_tensor(buf173, (1576, 3072), (3072, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (1576, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 3072), (1, 768), 0), out=buf191)
        del arg120_1
        buf192 = reinterpret_tensor(buf191, (8, 197, 3072), (605184, 3072, 1), 0); del buf191  # reuse
        # Source Nodes: [x_123], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf192, arg121_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg121_1
        buf193 = reinterpret_tensor(buf190, (1576, 768), (768, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg122_1, (3072, 768), (1, 3072), 0), out=buf193)
        del arg122_1
        buf197 = reinterpret_tensor(buf174, (8, 197, 768), (151296, 768, 1), 0); del buf174  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm1, x_128], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf186, buf193, arg123_1, arg124_1, arg125_1, buf197, 1576, 768, grid=grid(1576), stream=stream0)
        del arg124_1
        del arg125_1
        buf198 = buf179; del buf179  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg127_1, reinterpret_tensor(buf197, (1576, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf198)
        del arg126_1
        del arg127_1
        # Source Nodes: [x_129], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf199 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf200 = buf199[0]
        del buf199
        buf204 = reinterpret_tensor(buf197, (1576, 768), (768, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (1576, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0), out=buf204)
        del arg128_1
        buf205 = reinterpret_tensor(buf204, (8, 197, 768), (151296, 768, 1), 0); del buf204  # reuse
        buf209 = reinterpret_tensor(buf200, (8, 197, 768), (151296, 768, 1), 0); del buf200  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_128, x_133], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf205, buf186, buf193, arg123_1, arg129_1, arg130_1, arg131_1, buf209, 1576, 768, grid=grid(1576), stream=stream0)
        del arg123_1
        del arg129_1
        del arg130_1
        del arg131_1
        del buf186
        buf210 = reinterpret_tensor(buf192, (1576, 3072), (3072, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (1576, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 3072), (1, 768), 0), out=buf210)
        del arg132_1
        buf211 = reinterpret_tensor(buf210, (8, 197, 3072), (605184, 3072, 1), 0); del buf210  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf211, arg133_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg133_1
        buf212 = reinterpret_tensor(buf209, (1576, 768), (768, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg134_1, (3072, 768), (1, 3072), 0), out=buf212)
        del arg134_1
        buf216 = reinterpret_tensor(buf193, (8, 197, 768), (151296, 768, 1), 0); del buf193  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm1, x_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf205, buf212, arg135_1, arg136_1, arg137_1, buf216, 1576, 768, grid=grid(1576), stream=stream0)
        del arg136_1
        del arg137_1
        buf217 = buf198; del buf198  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg139_1, reinterpret_tensor(buf216, (1576, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf217)
        del arg138_1
        del arg139_1
        # Source Nodes: [x_141], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf218 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf217, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf217, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf217, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        del buf217
        buf219 = buf218[0]
        del buf218
        buf223 = reinterpret_tensor(buf216, (1576, 768), (768, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1576, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), out=buf223)
        del arg140_1
        buf224 = reinterpret_tensor(buf223, (8, 197, 768), (151296, 768, 1), 0); del buf223  # reuse
        buf228 = reinterpret_tensor(buf219, (8, 197, 768), (151296, 768, 1), 0); del buf219  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_140, x_145], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf224, buf205, buf212, arg135_1, arg141_1, arg142_1, arg143_1, buf228, 1576, 768, grid=grid(1576), stream=stream0)
        del arg135_1
        del arg141_1
        del arg142_1
        del arg143_1
        del buf205
        del buf212
        buf229 = reinterpret_tensor(buf211, (1576, 3072), (3072, 1), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (1576, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 3072), (1, 768), 0), out=buf229)
        del arg144_1
        buf230 = reinterpret_tensor(buf229, (8, 197, 3072), (605184, 3072, 1), 0); del buf229  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf230, arg145_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg145_1
        buf231 = reinterpret_tensor(buf228, (1576, 768), (768, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg146_1, (3072, 768), (1, 3072), 0), out=buf231)
        del arg146_1
        del buf230
        buf232 = buf5; del buf5  # reuse
        buf233 = buf4; del buf4  # reuse
        # Source Nodes: [x_153, x_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf224, buf231, arg147_1, buf232, buf233, 1576, 768, grid=grid(1576), stream=stream0)
        buf235 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf224, buf231, arg147_1, buf232, buf233, arg148_1, arg149_1, buf235, 6144, grid=grid(6144), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del buf224
        del buf231
        del buf232
        del buf233
        buf236 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158, x_159], Original ATen: [aten.addmm, aten.clone]
        extern_kernels.addmm(arg151_1, buf235, reinterpret_tensor(arg150_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf236)
        del arg150_1
        del arg151_1
        return (buf236, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vit_base_patch16_224', benchmark_compiled_module)
