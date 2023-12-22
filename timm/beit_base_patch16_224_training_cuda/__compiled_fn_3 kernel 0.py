
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


# kernel path: /tmp/torchinductor_youkaichao/pa/cpajdnvvr2ujplw2scfekzo3errb6svjqlo77uhcaemtrbwzqfph.py
# Source Nodes: [cat_25, qkv, x_6], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
# cat_25 => cat
# qkv => view_1
# x_6 => add, add_1, mul, mul_1, rsqrt, sub, var_mean
triton_per_fused_cat_native_layer_norm_view_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_0', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp40 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 768, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = 768.0
    tmp34 = tmp32 / tmp33
    tmp35 = 1e-06
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp16 - tmp26
    tmp39 = tmp38 * tmp37
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp16, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp37, xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp26, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/so/csol6g3m7iazxr6duog7736hmgbrsx3ddk3hawq4leagawy7rl76.py
# Source Nodes: [cat_24], Original ATen: [aten.cat]
# cat_24 => cat_1
triton_poi_fused_cat_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7exsauqbjxg5gmrf7rbramkpndltwp3fhkke6i5z42x7jv4vzz.py
# Source Nodes: [x_7], Original ATen: [aten.constant_pad_nd]
# x_7 => constant_pad_nd
triton_poi_fused_constant_pad_nd_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_2', 'mutated_arg_names': []},
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
    tl.device_assert(((0 <= tmp6) & (tmp6 < 732)) | ~(tmp2 & xmask), "index out of bounds: 0 <= tmp6 < 732")
    tmp7 = tl.load(in_ptr1 + (x2 + (12*tmp6)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjunyiehue2aguccuntqkpirx6oqc2bh63uht6r6fvwqlcmmohyv.py
# Source Nodes: [mul, x_11, x_12, x_13], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# mul => mul_2
# x_11 => add_2
# x_12 => add_3, add_4, mul_3, mul_4, rsqrt_1, sub_1, var_mean_1
# x_13 => view_9
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3', 'mutated_arg_names': []}
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
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


# kernel path: /tmp/torchinductor_youkaichao/md/cmdowuu34wy72sslbwkqoi6pe7waaj7bbdtapxyzsvo5gcg2t6ub.py
# Source Nodes: [x_14, x_17], Original ATen: [aten.gelu, aten.view]
# x_14 => add_5, erf, mul_5, mul_6, mul_7
# x_17 => view_11
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


# kernel path: /tmp/torchinductor_youkaichao/kl/ckl44cpd5q2r54azwhx6fpjlek57wv4ymuvtgkyu2hxdnenow4nh.py
# Source Nodes: [mul, mul_1, qkv_2, x_11, x_20, x_21], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# mul => mul_2
# mul_1 => mul_8
# qkv_2 => view_13
# x_11 => add_2
# x_20 => add_6
# x_21 => add_7, add_8, mul_10, mul_9, rsqrt_2, sub_2, var_mean_2
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
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
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmvuiyetgdi5o7p6i22zbk222z6on72sfffvjnuhtv642hv5mwp.py
# Source Nodes: [x_188], Original ATen: [aten.mean]
# x_188 => mean
triton_red_fused_mean_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (768 + x0 + (768*r3) + (75264*x1) + (151296*x2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (768 + x0 + (768*r3) + (75264*x1) + (151296*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr4 + (768 + x0 + (768*r3) + (75264*x1) + (151296*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zc/czcntb7jd75apslgykqzehbtqisn2jfihbqgogyo3p7getqn2b3x.py
# Source Nodes: [x_188], Original ATen: [aten.mean]
# x_188 => mean
triton_per_fused_mean_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ll/cll2bj7gglg2wvlnbyhhkgwp76b26uxch37fh3di3cadmcnzotqb.py
# Source Nodes: [x_188, x_190], Original ATen: [aten.mean, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_188 => mean
# x_190 => add_84, add_85, mul_108, mul_109, rsqrt_24, sub_24, var_mean_24
triton_per_fused_mean_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (2304, 768), (768, 1))
    assert_size_stride(primals_8, (732, 12), (12, 1))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (2304, 768), (768, 1))
    assert_size_stride(primals_18, (732, 12), (12, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (2304, 768), (768, 1))
    assert_size_stride(primals_28, (732, 12), (12, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (2304, 768), (768, 1))
    assert_size_stride(primals_38, (732, 12), (12, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (2304, 768), (768, 1))
    assert_size_stride(primals_48, (732, 12), (12, 1))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (2304, 768), (768, 1))
    assert_size_stride(primals_58, (732, 12), (12, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (2304, 768), (768, 1))
    assert_size_stride(primals_68, (732, 12), (12, 1))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (2304, 768), (768, 1))
    assert_size_stride(primals_78, (732, 12), (12, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (2304, 768), (768, 1))
    assert_size_stride(primals_88, (732, 12), (12, 1))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (2304, 768), (768, 1))
    assert_size_stride(primals_98, (732, 12), (12, 1))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (2304, 768), (768, 1))
    assert_size_stride(primals_108, (732, 12), (12, 1))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (2304, 768), (768, 1))
    assert_size_stride(primals_118, (732, 12), (12, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, 768), (768, 1))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (3072, 768), (768, 1))
    assert_size_stride(primals_129, (3072, ), (1, ))
    assert_size_stride(primals_130, (768, 3072), (3072, 1))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, 768), (768, 1))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (3072, 768), (768, 1))
    assert_size_stride(primals_135, (3072, ), (1, ))
    assert_size_stride(primals_136, (768, 3072), (3072, 1))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, 768), (768, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (3072, 768), (768, 1))
    assert_size_stride(primals_141, (3072, ), (1, ))
    assert_size_stride(primals_142, (768, 3072), (3072, 1))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, 768), (768, 1))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_146, (3072, 768), (768, 1))
    assert_size_stride(primals_147, (3072, ), (1, ))
    assert_size_stride(primals_148, (768, 3072), (3072, 1))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (768, 768), (768, 1))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_152, (3072, 768), (768, 1))
    assert_size_stride(primals_153, (3072, ), (1, ))
    assert_size_stride(primals_154, (768, 3072), (3072, 1))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (768, 768), (768, 1))
    assert_size_stride(primals_157, (768, ), (1, ))
    assert_size_stride(primals_158, (3072, 768), (768, 1))
    assert_size_stride(primals_159, (3072, ), (1, ))
    assert_size_stride(primals_160, (768, 3072), (3072, 1))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (768, 768), (768, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (3072, 768), (768, 1))
    assert_size_stride(primals_165, (3072, ), (1, ))
    assert_size_stride(primals_166, (768, 3072), (3072, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (768, 768), (768, 1))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (3072, 768), (768, 1))
    assert_size_stride(primals_171, (3072, ), (1, ))
    assert_size_stride(primals_172, (768, 3072), (3072, 1))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (768, 768), (768, 1))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (3072, 768), (768, 1))
    assert_size_stride(primals_177, (3072, ), (1, ))
    assert_size_stride(primals_178, (768, 3072), (3072, 1))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, 768), (768, 1))
    assert_size_stride(primals_181, (768, ), (1, ))
    assert_size_stride(primals_182, (3072, 768), (768, 1))
    assert_size_stride(primals_183, (3072, ), (1, ))
    assert_size_stride(primals_184, (768, 3072), (3072, 1))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, 768), (768, 1))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (3072, 768), (768, 1))
    assert_size_stride(primals_189, (3072, ), (1, ))
    assert_size_stride(primals_190, (768, 3072), (3072, 1))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, 768), (768, 1))
    assert_size_stride(primals_193, (768, ), (1, ))
    assert_size_stride(primals_194, (3072, 768), (768, 1))
    assert_size_stride(primals_195, (3072, ), (1, ))
    assert_size_stride(primals_196, (768, 3072), (3072, 1))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (1000, 768), (768, 1))
    assert_size_stride(primals_199, (1000, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (197, 197), (197, 1))
    assert_size_stride(primals_202, (768, ), (1, ))
    assert_size_stride(primals_203, (197, 197), (197, 1))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (197, 197), (197, 1))
    assert_size_stride(primals_206, (768, ), (1, ))
    assert_size_stride(primals_207, (197, 197), (197, 1))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (197, 197), (197, 1))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (197, 197), (197, 1))
    assert_size_stride(primals_212, (768, ), (1, ))
    assert_size_stride(primals_213, (197, 197), (197, 1))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (197, 197), (197, 1))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (197, 197), (197, 1))
    assert_size_stride(primals_218, (768, ), (1, ))
    assert_size_stride(primals_219, (197, 197), (197, 1))
    assert_size_stride(primals_220, (768, ), (1, ))
    assert_size_stride(primals_221, (197, 197), (197, 1))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (197, 197), (197, 1))
    assert_size_stride(primals_224, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_224, primals_124, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        buf1 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf2 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf5 = reinterpret_tensor(buf3, (8, 197, 1), (197, 1, 1), 0); del buf3  # reuse
        buf6 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25, qkv, x_6], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_cat_native_layer_norm_view_0.run(buf5, primals_1, buf0, primals_125, primals_3, primals_4, buf1, buf2, buf6, 1576, 768, grid=grid(1576), stream=stream0)
        del buf0
        del primals_1
        del primals_125
        del primals_4
        buf7 = empty((2304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_5, primals_200, primals_6, buf7, 2304, grid=grid(2304), stream=stream0)
        del primals_200
        del primals_5
        del primals_6
        buf8 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24, qkv], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf7, buf6, reinterpret_tensor(primals_7, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf8)
        buf9 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_201, primals_8, buf9, 472800, grid=grid(472800), stream=stream0)
        del primals_8
        # Source Nodes: [x_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf10 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf9, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf11 = buf10[0]
        buf12 = buf10[1]
        buf13 = buf10[2]
        buf14 = buf10[3]
        del buf10
        buf15 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_127, reinterpret_tensor(buf11, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_126, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf15)
        del primals_127
        buf19 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf20 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf308 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul, x_11, x_12, x_13], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf1, primals_2, buf15, primals_10, primals_11, buf19, buf20, buf308, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_11
        buf21 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf20, reinterpret_tensor(primals_128, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf21)
        del primals_129
        buf22 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14, x_17], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf21, buf22, 4841472, grid=grid(4841472), stream=stream0)
        buf23 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_131, buf22, reinterpret_tensor(primals_130, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf23)
        del primals_131
        buf24 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf28 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf29 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf307 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul, mul_1, qkv_2, x_11, x_20, x_21], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf1, primals_2, buf15, primals_9, buf23, primals_13, primals_14, buf24, buf28, buf29, buf307, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_14
        buf30 = buf7; del buf7  # reuse
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_15, primals_202, primals_16, buf30, 2304, grid=grid(2304), stream=stream0)
        del primals_15
        del primals_16
        del primals_202
        buf31 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23, qkv_2], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf30, buf29, reinterpret_tensor(primals_17, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf31)
        buf32 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_203, primals_18, buf32, 472800, grid=grid(472800), stream=stream0)
        del primals_18
        # Source Nodes: [x_22], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf33 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf31, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf31, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf31, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf32, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf34 = buf33[0]
        buf35 = buf33[1]
        buf36 = buf33[2]
        buf37 = buf33[3]
        del buf33
        buf38 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_133, reinterpret_tensor(buf34, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_132, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf38)
        del primals_133
        buf42 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf43 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf306 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_2, x_26, x_27, x_28], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf24, primals_12, buf38, primals_20, primals_21, buf42, buf43, buf306, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_21
        buf44 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_135, buf43, reinterpret_tensor(primals_134, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf44)
        del primals_135
        buf45 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29, x_32], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf44, buf45, 4841472, grid=grid(4841472), stream=stream0)
        buf46 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_137, buf45, reinterpret_tensor(primals_136, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf46)
        del primals_137
        buf47 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf51 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf52 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf305 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_2, mul_3, qkv_4, x_26, x_35, x_36], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf24, primals_12, buf38, primals_19, buf46, primals_23, primals_24, buf47, buf51, buf52, buf305, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_24
        buf53 = buf30; del buf30  # reuse
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_25, primals_204, primals_26, buf53, 2304, grid=grid(2304), stream=stream0)
        del primals_204
        del primals_25
        del primals_26
        buf54 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_22, qkv_4], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf53, buf52, reinterpret_tensor(primals_27, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf54)
        buf55 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_205, primals_28, buf55, 472800, grid=grid(472800), stream=stream0)
        del primals_28
        # Source Nodes: [x_37], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf56 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf54, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf54, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf54, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf55, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf57 = buf56[0]
        buf58 = buf56[1]
        buf59 = buf56[2]
        buf60 = buf56[3]
        del buf56
        buf61 = reinterpret_tensor(buf24, (1576, 768), (768, 1), 0); del buf24  # reuse
        # Source Nodes: [x_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_139, reinterpret_tensor(buf57, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf61)
        del primals_139
        buf65 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf66 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf304 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_4, x_41, x_42, x_43], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf47, primals_22, buf61, primals_30, primals_31, buf65, buf66, buf304, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_31
        buf67 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_141, buf66, reinterpret_tensor(primals_140, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf67)
        del primals_141
        buf68 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44, x_47], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf67, buf68, 4841472, grid=grid(4841472), stream=stream0)
        buf69 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_143, buf68, reinterpret_tensor(primals_142, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf69)
        del primals_143
        buf70 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf74 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf75 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf303 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_4, mul_5, qkv_6, x_41, x_50, x_51], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf47, primals_22, buf61, primals_29, buf69, primals_33, primals_34, buf70, buf74, buf75, buf303, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_34
        buf76 = buf53; del buf53  # reuse
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_35, primals_206, primals_36, buf76, 2304, grid=grid(2304), stream=stream0)
        del primals_206
        del primals_35
        del primals_36
        buf77 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_21, qkv_6], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf76, buf75, reinterpret_tensor(primals_37, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf77)
        buf78 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_207, primals_38, buf78, 472800, grid=grid(472800), stream=stream0)
        del primals_38
        # Source Nodes: [x_52], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf79 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf77, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf77, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf77, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf78, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf80 = buf79[0]
        buf81 = buf79[1]
        buf82 = buf79[2]
        buf83 = buf79[3]
        del buf79
        buf84 = reinterpret_tensor(buf47, (1576, 768), (768, 1), 0); del buf47  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, reinterpret_tensor(buf80, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_144, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf84)
        del primals_145
        buf88 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf89 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf302 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_6, x_56, x_57, x_58], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf70, primals_32, buf84, primals_40, primals_41, buf88, buf89, buf302, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_41
        buf90 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_147, buf89, reinterpret_tensor(primals_146, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf90)
        del primals_147
        buf91 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59, x_62], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf90, buf91, 4841472, grid=grid(4841472), stream=stream0)
        buf92 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_149, buf91, reinterpret_tensor(primals_148, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf92)
        del primals_149
        buf93 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf97 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf98 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf301 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_6, mul_7, qkv_8, x_56, x_65, x_66], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf70, primals_32, buf84, primals_39, buf92, primals_43, primals_44, buf93, buf97, buf98, buf301, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_44
        buf99 = buf76; del buf76  # reuse
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_45, primals_208, primals_46, buf99, 2304, grid=grid(2304), stream=stream0)
        del primals_208
        del primals_45
        del primals_46
        buf100 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_20, qkv_8], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf99, buf98, reinterpret_tensor(primals_47, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf100)
        buf101 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_209, primals_48, buf101, 472800, grid=grid(472800), stream=stream0)
        del primals_48
        # Source Nodes: [x_67], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf102 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf100, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf100, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf100, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf101, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf103 = buf102[0]
        buf104 = buf102[1]
        buf105 = buf102[2]
        buf106 = buf102[3]
        del buf102
        buf107 = reinterpret_tensor(buf70, (1576, 768), (768, 1), 0); del buf70  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_151, reinterpret_tensor(buf103, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_150, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf107)
        del primals_151
        buf111 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf112 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf300 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_8, x_71, x_72, x_73], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf93, primals_42, buf107, primals_50, primals_51, buf111, buf112, buf300, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_51
        buf113 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_153, buf112, reinterpret_tensor(primals_152, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf113)
        del primals_153
        buf114 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_77], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf113, buf114, 4841472, grid=grid(4841472), stream=stream0)
        buf115 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_155, buf114, reinterpret_tensor(primals_154, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf115)
        del primals_155
        buf116 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf120 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf121 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf299 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_8, mul_9, qkv_10, x_71, x_80, x_81], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf93, primals_42, buf107, primals_49, buf115, primals_53, primals_54, buf116, buf120, buf121, buf299, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_54
        buf122 = buf99; del buf99  # reuse
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_55, primals_210, primals_56, buf122, 2304, grid=grid(2304), stream=stream0)
        del primals_210
        del primals_55
        del primals_56
        buf123 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_19, qkv_10], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf122, buf121, reinterpret_tensor(primals_57, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf123)
        buf124 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_211, primals_58, buf124, 472800, grid=grid(472800), stream=stream0)
        del primals_58
        # Source Nodes: [x_82], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf125 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf124, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf126 = buf125[0]
        buf127 = buf125[1]
        buf128 = buf125[2]
        buf129 = buf125[3]
        del buf125
        buf130 = reinterpret_tensor(buf93, (1576, 768), (768, 1), 0); del buf93  # reuse
        # Source Nodes: [x_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_157, reinterpret_tensor(buf126, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_156, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf130)
        del primals_157
        buf134 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf135 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf298 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_10, x_86, x_87, x_88], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf116, primals_52, buf130, primals_60, primals_61, buf134, buf135, buf298, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_61
        buf136 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_159, buf135, reinterpret_tensor(primals_158, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf136)
        del primals_159
        buf137 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89, x_92], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf136, buf137, 4841472, grid=grid(4841472), stream=stream0)
        buf138 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf137, reinterpret_tensor(primals_160, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf138)
        del primals_161
        buf139 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf143 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf144 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf297 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_10, mul_11, qkv_12, x_86, x_95, x_96], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf116, primals_52, buf130, primals_59, buf138, primals_63, primals_64, buf139, buf143, buf144, buf297, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_64
        buf145 = buf122; del buf122  # reuse
        # Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_65, primals_212, primals_66, buf145, 2304, grid=grid(2304), stream=stream0)
        del primals_212
        del primals_65
        del primals_66
        buf146 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_18, qkv_12], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf145, buf144, reinterpret_tensor(primals_67, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf146)
        buf147 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_213, primals_68, buf147, 472800, grid=grid(472800), stream=stream0)
        del primals_68
        # Source Nodes: [x_97], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf148 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf146, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf146, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf146, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf147, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf149 = buf148[0]
        buf150 = buf148[1]
        buf151 = buf148[2]
        buf152 = buf148[3]
        del buf148
        buf153 = reinterpret_tensor(buf116, (1576, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_163, reinterpret_tensor(buf149, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_162, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf153)
        del primals_163
        buf157 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf158 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf296 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_12, x_101, x_102, x_103], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf139, primals_62, buf153, primals_70, primals_71, buf157, buf158, buf296, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_71
        buf159 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_165, buf158, reinterpret_tensor(primals_164, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf159)
        del primals_165
        buf160 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104, x_107], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf159, buf160, 4841472, grid=grid(4841472), stream=stream0)
        buf161 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_167, buf160, reinterpret_tensor(primals_166, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf161)
        del primals_167
        buf162 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf166 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf167 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf295 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_12, mul_13, qkv_14, x_101, x_110, x_111], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf139, primals_62, buf153, primals_69, buf161, primals_73, primals_74, buf162, buf166, buf167, buf295, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_74
        buf168 = buf145; del buf145  # reuse
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_75, primals_214, primals_76, buf168, 2304, grid=grid(2304), stream=stream0)
        del primals_214
        del primals_75
        del primals_76
        buf169 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_17, qkv_14], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf168, buf167, reinterpret_tensor(primals_77, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf169)
        buf170 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_215, primals_78, buf170, 472800, grid=grid(472800), stream=stream0)
        del primals_78
        # Source Nodes: [x_112], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf171 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf169, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf169, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf169, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf170, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf172 = buf171[0]
        buf173 = buf171[1]
        buf174 = buf171[2]
        buf175 = buf171[3]
        del buf171
        buf176 = reinterpret_tensor(buf139, (1576, 768), (768, 1), 0); del buf139  # reuse
        # Source Nodes: [x_114], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_169, reinterpret_tensor(buf172, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_168, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf176)
        del primals_169
        buf180 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf181 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf294 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_14, x_116, x_117, x_118], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf162, primals_72, buf176, primals_80, primals_81, buf180, buf181, buf294, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_81
        buf182 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_171, buf181, reinterpret_tensor(primals_170, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf182)
        del primals_171
        buf183 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_122], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf182, buf183, 4841472, grid=grid(4841472), stream=stream0)
        buf184 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_173, buf183, reinterpret_tensor(primals_172, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf184)
        del primals_173
        buf185 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf189 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf190 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf293 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_14, mul_15, qkv_16, x_116, x_125, x_126], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf162, primals_72, buf176, primals_79, buf184, primals_83, primals_84, buf185, buf189, buf190, buf293, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_84
        buf191 = buf168; del buf168  # reuse
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_85, primals_216, primals_86, buf191, 2304, grid=grid(2304), stream=stream0)
        del primals_216
        del primals_85
        del primals_86
        buf192 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_16, qkv_16], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf191, buf190, reinterpret_tensor(primals_87, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf192)
        buf193 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_217, primals_88, buf193, 472800, grid=grid(472800), stream=stream0)
        del primals_88
        # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf194 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf192, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf192, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf192, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf193, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf195 = buf194[0]
        buf196 = buf194[1]
        buf197 = buf194[2]
        buf198 = buf194[3]
        del buf194
        buf199 = reinterpret_tensor(buf162, (1576, 768), (768, 1), 0); del buf162  # reuse
        # Source Nodes: [x_129], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_175, reinterpret_tensor(buf195, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_174, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf199)
        del primals_175
        buf203 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf204 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf292 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_16, x_131, x_132, x_133], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf185, primals_82, buf199, primals_90, primals_91, buf203, buf204, buf292, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_91
        buf205 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf204, reinterpret_tensor(primals_176, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf205)
        del primals_177
        buf206 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134, x_137], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf205, buf206, 4841472, grid=grid(4841472), stream=stream0)
        buf207 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_179, buf206, reinterpret_tensor(primals_178, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf207)
        del primals_179
        buf208 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf212 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf213 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf291 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_16, mul_17, qkv_18, x_131, x_140, x_141], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf185, primals_82, buf199, primals_89, buf207, primals_93, primals_94, buf208, buf212, buf213, buf291, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_94
        buf214 = buf191; del buf191  # reuse
        # Source Nodes: [cat_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_95, primals_218, primals_96, buf214, 2304, grid=grid(2304), stream=stream0)
        del primals_218
        del primals_95
        del primals_96
        buf215 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_15, qkv_18], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf214, buf213, reinterpret_tensor(primals_97, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf215)
        buf216 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_219, primals_98, buf216, 472800, grid=grid(472800), stream=stream0)
        del primals_98
        # Source Nodes: [x_142], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf217 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf215, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf215, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf215, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf216, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf218 = buf217[0]
        buf219 = buf217[1]
        buf220 = buf217[2]
        buf221 = buf217[3]
        del buf217
        buf222 = reinterpret_tensor(buf185, (1576, 768), (768, 1), 0); del buf185  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_181, reinterpret_tensor(buf218, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_180, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf222)
        del primals_181
        buf226 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf227 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf290 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_18, x_146, x_147, x_148], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf208, primals_92, buf222, primals_100, primals_101, buf226, buf227, buf290, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_101
        buf228 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_183, buf227, reinterpret_tensor(primals_182, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf228)
        del primals_183
        buf229 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149, x_152], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf228, buf229, 4841472, grid=grid(4841472), stream=stream0)
        buf230 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_185, buf229, reinterpret_tensor(primals_184, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf230)
        del primals_185
        buf231 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf235 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf236 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf289 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_18, mul_19, qkv_20, x_146, x_155, x_156], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf208, primals_92, buf222, primals_99, buf230, primals_103, primals_104, buf231, buf235, buf236, buf289, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_104
        buf237 = buf214; del buf214  # reuse
        # Source Nodes: [cat_14], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_105, primals_220, primals_106, buf237, 2304, grid=grid(2304), stream=stream0)
        del primals_105
        del primals_106
        del primals_220
        buf238 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_14, qkv_20], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf237, buf236, reinterpret_tensor(primals_107, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf238)
        buf239 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_221, primals_108, buf239, 472800, grid=grid(472800), stream=stream0)
        del primals_108
        # Source Nodes: [x_157], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf240 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf238, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf238, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf238, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf239, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf241 = buf240[0]
        buf242 = buf240[1]
        buf243 = buf240[2]
        buf244 = buf240[3]
        del buf240
        buf245 = reinterpret_tensor(buf208, (1576, 768), (768, 1), 0); del buf208  # reuse
        # Source Nodes: [x_159], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_187, reinterpret_tensor(buf241, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_186, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf245)
        del primals_187
        buf249 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf250 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf288 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_20, x_161, x_162, x_163], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf231, primals_102, buf245, primals_110, primals_111, buf249, buf250, buf288, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_111
        buf251 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_189, buf250, reinterpret_tensor(primals_188, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf251)
        del primals_189
        buf252 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164, x_167], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf251, buf252, 4841472, grid=grid(4841472), stream=stream0)
        buf253 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_191, buf252, reinterpret_tensor(primals_190, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf253)
        del primals_191
        buf254 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf258 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf259 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf287 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_20, mul_21, qkv_22, x_161, x_170, x_171], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_5.run(buf231, primals_102, buf245, primals_109, buf253, primals_113, primals_114, buf254, buf258, buf259, buf287, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_114
        buf260 = buf237; del buf237  # reuse
        # Source Nodes: [cat_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(primals_115, primals_222, primals_116, buf260, 2304, grid=grid(2304), stream=stream0)
        del primals_115
        del primals_116
        del primals_222
        buf261 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_13, qkv_22], Original ATen: [aten.addmm, aten.cat]
        extern_kernels.addmm(buf260, buf259, reinterpret_tensor(primals_117, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf261)
        del buf260
        buf262 = empty((1, 12, 197, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_2.run(primals_223, primals_118, buf262, 472800, grid=grid(472800), stream=stream0)
        del primals_118
        # Source Nodes: [x_172], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf263 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf261, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf261, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf261, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(buf262, (8, 12, 197, 197), (0, 39400, 200, 1), 0), True)
        buf264 = buf263[0]
        buf265 = buf263[1]
        buf266 = buf263[2]
        buf267 = buf263[3]
        del buf263
        buf268 = reinterpret_tensor(buf231, (1576, 768), (768, 1), 0); del buf231  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_193, reinterpret_tensor(buf264, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_192, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf268)
        del primals_193
        buf272 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf273 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf286 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_22, x_176, x_177, x_178], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_3.run(buf254, primals_112, buf268, primals_120, primals_121, buf272, buf273, buf286, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_121
        buf274 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_195, buf273, reinterpret_tensor(primals_194, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf274)
        del primals_195
        buf275 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179, x_182], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf274, buf275, 4841472, grid=grid(4841472), stream=stream0)
        buf276 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_197, buf275, reinterpret_tensor(primals_196, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf276)
        del primals_197
        buf277 = empty_strided((8, 768, 2), (1536, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.mean]
        triton_red_fused_mean_6.run(buf254, primals_112, buf268, primals_119, buf276, buf277, 12288, 98, grid=grid(12288), stream=stream0)
        del buf254
        buf278 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.mean]
        triton_per_fused_mean_7.run(buf277, buf278, 6144, 2, grid=grid(6144), stream=stream0)
        del buf277
        buf282 = empty((8, 768), device='cuda', dtype=torch.float32)
        buf283 = empty((8, 768), device='cuda', dtype=torch.float32)
        buf285 = empty((8, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188, x_190], Original ATen: [aten.mean, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_mean_native_layer_norm_native_layer_norm_backward_8.run(buf278, primals_122, primals_123, buf282, buf283, buf285, 8, 768, grid=grid(8), stream=stream0)
        del buf278
        del primals_123
        buf284 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_199, buf283, reinterpret_tensor(primals_198, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf284)
        del primals_199
        return (buf284, primals_2, primals_3, primals_9, primals_10, primals_12, primals_13, primals_19, primals_20, primals_22, primals_23, primals_29, primals_30, primals_32, primals_33, primals_39, primals_40, primals_42, primals_43, primals_49, primals_50, primals_52, primals_53, primals_59, primals_60, primals_62, primals_63, primals_69, primals_70, primals_72, primals_73, primals_79, primals_80, primals_82, primals_83, primals_89, primals_90, primals_92, primals_93, primals_99, primals_100, primals_102, primals_103, primals_109, primals_110, primals_112, primals_113, primals_119, primals_120, primals_122, primals_124, primals_224, buf1, buf2, buf5, buf6, reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf8, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_201, (38809, ), (1, ), 0), reinterpret_tensor(buf9, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf12, buf13, buf14, reinterpret_tensor(buf11, (1576, 768), (768, 1), 0), buf15, buf19, buf20, buf21, buf22, buf23, buf28, buf29, reinterpret_tensor(buf31, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf31, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf31, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_203, (38809, ), (1, ), 0), reinterpret_tensor(buf32, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf35, buf36, buf37, reinterpret_tensor(buf34, (1576, 768), (768, 1), 0), buf38, buf42, buf43, buf44, buf45, buf46, buf51, buf52, reinterpret_tensor(buf54, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf54, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf54, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_205, (38809, ), (1, ), 0), reinterpret_tensor(buf55, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf58, buf59, buf60, reinterpret_tensor(buf57, (1576, 768), (768, 1), 0), buf61, buf65, buf66, buf67, buf68, buf69, buf74, buf75, reinterpret_tensor(buf77, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf77, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf77, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_207, (38809, ), (1, ), 0), reinterpret_tensor(buf78, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf81, buf82, buf83, reinterpret_tensor(buf80, (1576, 768), (768, 1), 0), buf84, buf88, buf89, buf90, buf91, buf92, buf97, buf98, reinterpret_tensor(buf100, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf100, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf100, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_209, (38809, ), (1, ), 0), reinterpret_tensor(buf101, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf104, buf105, buf106, reinterpret_tensor(buf103, (1576, 768), (768, 1), 0), buf107, buf111, buf112, buf113, buf114, buf115, buf120, buf121, reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_211, (38809, ), (1, ), 0), reinterpret_tensor(buf124, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf127, buf128, buf129, reinterpret_tensor(buf126, (1576, 768), (768, 1), 0), buf130, buf134, buf135, buf136, buf137, buf138, buf143, buf144, reinterpret_tensor(buf146, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf146, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf146, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_213, (38809, ), (1, ), 0), reinterpret_tensor(buf147, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf150, buf151, buf152, reinterpret_tensor(buf149, (1576, 768), (768, 1), 0), buf153, buf157, buf158, buf159, buf160, buf161, buf166, buf167, reinterpret_tensor(buf169, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf169, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf169, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_215, (38809, ), (1, ), 0), reinterpret_tensor(buf170, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf173, buf174, buf175, reinterpret_tensor(buf172, (1576, 768), (768, 1), 0), buf176, buf180, buf181, buf182, buf183, buf184, buf189, buf190, reinterpret_tensor(buf192, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf192, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf192, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_217, (38809, ), (1, ), 0), reinterpret_tensor(buf193, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf196, buf197, buf198, reinterpret_tensor(buf195, (1576, 768), (768, 1), 0), buf199, buf203, buf204, buf205, buf206, buf207, buf212, buf213, reinterpret_tensor(buf215, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf215, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf215, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_219, (38809, ), (1, ), 0), reinterpret_tensor(buf216, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf219, buf220, buf221, reinterpret_tensor(buf218, (1576, 768), (768, 1), 0), buf222, buf226, buf227, buf228, buf229, buf230, buf235, buf236, reinterpret_tensor(buf238, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf238, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf238, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_221, (38809, ), (1, ), 0), reinterpret_tensor(buf239, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf242, buf243, buf244, reinterpret_tensor(buf241, (1576, 768), (768, 1), 0), buf245, buf249, buf250, buf251, buf252, buf253, buf258, buf259, reinterpret_tensor(buf261, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf261, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf261, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), reinterpret_tensor(primals_223, (38809, ), (1, ), 0), reinterpret_tensor(buf262, (8, 12, 197, 197), (0, 39400, 200, 1), 0), buf265, buf266, buf267, reinterpret_tensor(buf264, (1576, 768), (768, 1), 0), buf268, buf272, buf273, buf274, buf275, buf276, buf282, buf283, reinterpret_tensor(primals_198, (1000, 768), (768, 1), 0), buf285, reinterpret_tensor(primals_196, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_194, (3072, 768), (768, 1), 0), buf286, reinterpret_tensor(primals_192, (768, 768), (768, 1), 0), buf264, reinterpret_tensor(primals_117, (2304, 768), (768, 1), 0), buf287, reinterpret_tensor(primals_190, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_188, (3072, 768), (768, 1), 0), buf288, reinterpret_tensor(primals_186, (768, 768), (768, 1), 0), buf241, reinterpret_tensor(primals_107, (2304, 768), (768, 1), 0), buf289, reinterpret_tensor(primals_184, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_182, (3072, 768), (768, 1), 0), buf290, reinterpret_tensor(primals_180, (768, 768), (768, 1), 0), buf218, reinterpret_tensor(primals_97, (2304, 768), (768, 1), 0), buf291, reinterpret_tensor(primals_178, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_176, (3072, 768), (768, 1), 0), buf292, reinterpret_tensor(primals_174, (768, 768), (768, 1), 0), buf195, reinterpret_tensor(primals_87, (2304, 768), (768, 1), 0), buf293, reinterpret_tensor(primals_172, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_170, (3072, 768), (768, 1), 0), buf294, reinterpret_tensor(primals_168, (768, 768), (768, 1), 0), buf172, reinterpret_tensor(primals_77, (2304, 768), (768, 1), 0), buf295, reinterpret_tensor(primals_166, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_164, (3072, 768), (768, 1), 0), buf296, reinterpret_tensor(primals_162, (768, 768), (768, 1), 0), buf149, reinterpret_tensor(primals_67, (2304, 768), (768, 1), 0), buf297, reinterpret_tensor(primals_160, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_158, (3072, 768), (768, 1), 0), buf298, reinterpret_tensor(primals_156, (768, 768), (768, 1), 0), buf126, reinterpret_tensor(primals_57, (2304, 768), (768, 1), 0), buf299, reinterpret_tensor(primals_154, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_152, (3072, 768), (768, 1), 0), buf300, reinterpret_tensor(primals_150, (768, 768), (768, 1), 0), buf103, reinterpret_tensor(primals_47, (2304, 768), (768, 1), 0), buf301, reinterpret_tensor(primals_148, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_146, (3072, 768), (768, 1), 0), buf302, reinterpret_tensor(primals_144, (768, 768), (768, 1), 0), buf80, reinterpret_tensor(primals_37, (2304, 768), (768, 1), 0), buf303, reinterpret_tensor(primals_142, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_140, (3072, 768), (768, 1), 0), buf304, reinterpret_tensor(primals_138, (768, 768), (768, 1), 0), buf57, reinterpret_tensor(primals_27, (2304, 768), (768, 1), 0), buf305, reinterpret_tensor(primals_136, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_134, (3072, 768), (768, 1), 0), buf306, reinterpret_tensor(primals_132, (768, 768), (768, 1), 0), buf34, reinterpret_tensor(primals_17, (2304, 768), (768, 1), 0), buf307, reinterpret_tensor(primals_130, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_128, (3072, 768), (768, 1), 0), buf308, reinterpret_tensor(primals_126, (768, 768), (768, 1), 0), buf11, reinterpret_tensor(primals_7, (2304, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((732, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_212 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_218 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((197, 197), (197, 1), device='cuda:0', dtype=torch.int64)
    primals_224 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('beit_base_patch16_224', benchmark_compiled_module)
