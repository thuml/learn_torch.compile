
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


# kernel path: /tmp/torchinductor_youkaichao/3f/c3flrih2zzpffvmsvl7xzmomrm6c7awi3czp7gcvmziycnuerug4.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___attn_qkv, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cat_1 => cat
# getattr_l__mod___blocks___0___attn_qkv => view_1
# getattr_l__mod___blocks___0___norm1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# x_5 => add
triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 1584
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 198
    r2 = rindex
    x1 = (xindex // 198)
    x3 = xindex
    tmp25 = tl.load(in_ptr4 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 198, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((196*r2) + (150528*x1) + (((-2) + x0) % 196)), rmask & tmp15 & xmask, other=0.0)
    tmp19 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tl.where(tmp11, tmp14, tmp22)
    tmp24 = tl.where(tmp4, tmp7, tmp23)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tl.full([1], 768, tl.int32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 / tmp35
    tmp37 = tmp27 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp43 = tmp26 - tmp36
    tmp44 = 768.0
    tmp45 = tmp42 / tmp44
    tmp46 = 1e-06
    tmp47 = tmp45 + tmp46
    tmp48 = tl.math.rsqrt(tmp47)
    tmp49 = tmp43 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tmp48 / tmp44
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp49, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp53, rmask & xmask)
    tl.store(out_ptr5 + (x3), tmp54, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnawtfapoziwmxx7rrnm5gm7mx3ycvoe3tormbtrsjculfd2ofy.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_13, x_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks___0___norm2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# x_13 => add_3
# x_14 => view_7
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1584
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


# kernel path: /tmp/torchinductor_youkaichao/66/c66ytwzpnt7nnrbwvkdlfc2ine7rhbdtc55qnjsdchtgd3ii4sib.py
# Source Nodes: [x_15, x_18], Original ATen: [aten.gelu, aten.view]
# x_15 => add_6, erf, mul_4, mul_5, mul_6
# x_18 => view_9
triton_poi_fused_gelu_view_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4866048
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


# kernel path: /tmp/torchinductor_youkaichao/xq/cxqbodvuhg2kfxhdqm7av63h24ybgsf3h2pz4rzuxcrrfpqw2eok.py
# Source Nodes: [getattr_l__mod___blocks___1___attn_qkv, getattr_l__mod___blocks___1___norm1, x_13, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks___1___attn_qkv => view_11
# getattr_l__mod___blocks___1___norm1 => add_8, add_9, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
# x_13 => add_3
# x_20 => add_7
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1584
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


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwnjn5dz45ekxecf5xvcgwi5ua6imwpilz5svyd2skh4ystnzcx.py
# Source Nodes: [add_25, pred], Original ATen: [aten.add, aten.div]
# add_25 => add_87
# pred => div
triton_poi_fused_add_div_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 2.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156 = args
    args.clear()
    assert_size_stride(primals_1, (1, 198, 768), (152064, 768, 1))
    assert_size_stride(primals_2, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_3, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_4, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (2304, 768), (768, 1))
    assert_size_stride(primals_9, (2304, ), (1, ))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (3072, 768), (768, 1))
    assert_size_stride(primals_15, (3072, ), (1, ))
    assert_size_stride(primals_16, (768, 3072), (3072, 1))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (2304, 768), (768, 1))
    assert_size_stride(primals_21, (2304, ), (1, ))
    assert_size_stride(primals_22, (768, 768), (768, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (3072, 768), (768, 1))
    assert_size_stride(primals_27, (3072, ), (1, ))
    assert_size_stride(primals_28, (768, 3072), (3072, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (2304, 768), (768, 1))
    assert_size_stride(primals_33, (2304, ), (1, ))
    assert_size_stride(primals_34, (768, 768), (768, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (3072, 768), (768, 1))
    assert_size_stride(primals_39, (3072, ), (1, ))
    assert_size_stride(primals_40, (768, 3072), (3072, 1))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (2304, 768), (768, 1))
    assert_size_stride(primals_45, (2304, ), (1, ))
    assert_size_stride(primals_46, (768, 768), (768, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (3072, 768), (768, 1))
    assert_size_stride(primals_51, (3072, ), (1, ))
    assert_size_stride(primals_52, (768, 3072), (3072, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (2304, 768), (768, 1))
    assert_size_stride(primals_57, (2304, ), (1, ))
    assert_size_stride(primals_58, (768, 768), (768, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (3072, 768), (768, 1))
    assert_size_stride(primals_63, (3072, ), (1, ))
    assert_size_stride(primals_64, (768, 3072), (3072, 1))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (2304, 768), (768, 1))
    assert_size_stride(primals_69, (2304, ), (1, ))
    assert_size_stride(primals_70, (768, 768), (768, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (3072, 768), (768, 1))
    assert_size_stride(primals_75, (3072, ), (1, ))
    assert_size_stride(primals_76, (768, 3072), (3072, 1))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (2304, 768), (768, 1))
    assert_size_stride(primals_81, (2304, ), (1, ))
    assert_size_stride(primals_82, (768, 768), (768, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (3072, 768), (768, 1))
    assert_size_stride(primals_87, (3072, ), (1, ))
    assert_size_stride(primals_88, (768, 3072), (3072, 1))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (2304, 768), (768, 1))
    assert_size_stride(primals_93, (2304, ), (1, ))
    assert_size_stride(primals_94, (768, 768), (768, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (3072, 768), (768, 1))
    assert_size_stride(primals_99, (3072, ), (1, ))
    assert_size_stride(primals_100, (768, 3072), (3072, 1))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (2304, 768), (768, 1))
    assert_size_stride(primals_105, (2304, ), (1, ))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (3072, 768), (768, 1))
    assert_size_stride(primals_111, (3072, ), (1, ))
    assert_size_stride(primals_112, (768, 3072), (3072, 1))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (2304, 768), (768, 1))
    assert_size_stride(primals_117, (2304, ), (1, ))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (3072, 768), (768, 1))
    assert_size_stride(primals_123, (3072, ), (1, ))
    assert_size_stride(primals_124, (768, 3072), (3072, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (2304, 768), (768, 1))
    assert_size_stride(primals_129, (2304, ), (1, ))
    assert_size_stride(primals_130, (768, 768), (768, 1))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (3072, 768), (768, 1))
    assert_size_stride(primals_135, (3072, ), (1, ))
    assert_size_stride(primals_136, (768, 3072), (3072, 1))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (2304, 768), (768, 1))
    assert_size_stride(primals_141, (2304, ), (1, ))
    assert_size_stride(primals_142, (768, 768), (768, 1))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_146, (3072, 768), (768, 1))
    assert_size_stride(primals_147, (3072, ), (1, ))
    assert_size_stride(primals_148, (768, 3072), (3072, 1))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_152, (1000, 768), (768, 1))
    assert_size_stride(primals_153, (1000, ), (1, ))
    assert_size_stride(primals_154, (1000, 768), (768, 1))
    assert_size_stride(primals_155, (1000, ), (1, ))
    assert_size_stride(primals_156, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_156, primals_4, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        buf1 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf6 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf286 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___attn_qkv, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_0.run(primals_2, primals_3, buf0, primals_5, primals_1, primals_6, primals_7, buf1, buf5, buf6, buf286, 1584, 768, grid=grid(1584), stream=stream0)
        del buf0
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        del primals_7
        buf7 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, buf6, reinterpret_tensor(primals_8, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del primals_9
        # Source Nodes: [x_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf8 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf7, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf7, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf7, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf9 = buf8[0]
        buf10 = buf8[1]
        buf11 = buf8[2]
        buf12 = buf8[3]
        del buf8
        buf13 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf13)
        buf17 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf18 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf285 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_13, x_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf1, buf13, primals_11, primals_12, primals_13, buf17, buf18, buf285, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_13
        buf19 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_15, buf18, reinterpret_tensor(primals_14, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf19)
        del primals_15
        buf20 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_15, x_18], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf19, buf20, 4866048, grid=grid(4866048), stream=stream0)
        buf21 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf20, reinterpret_tensor(primals_16, (3072, 768), (1, 3072), 0), out=buf21)
        buf22 = reinterpret_tensor(buf21, (8, 198, 768), (152064, 768, 1), 0); del buf21  # reuse
        buf26 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf27 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf284 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___attn_qkv, getattr_l__mod___blocks___1___norm1, x_13, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf22, buf1, buf13, primals_11, primals_17, primals_18, primals_19, buf26, buf27, buf284, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_11
        del primals_17
        del primals_19
        buf28 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, buf27, reinterpret_tensor(primals_20, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf28)
        del primals_21
        # Source Nodes: [x_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf29 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf28, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf28, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf28, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf30 = buf29[0]
        buf31 = buf29[1]
        buf32 = buf29[2]
        buf33 = buf29[3]
        del buf29
        buf34 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 768), (1, 768), 0), out=buf34)
        buf38 = buf1; del buf1  # reuse
        buf39 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf283 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_25, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf22, buf34, primals_23, primals_24, primals_25, buf38, buf39, buf283, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_25
        buf40 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_27, buf39, reinterpret_tensor(primals_26, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf40)
        del primals_27
        buf41 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27, x_30], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf40, buf41, 4866048, grid=grid(4866048), stream=stream0)
        buf42 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, reinterpret_tensor(primals_28, (3072, 768), (1, 3072), 0), out=buf42)
        buf43 = reinterpret_tensor(buf42, (8, 198, 768), (152064, 768, 1), 0); del buf42  # reuse
        buf47 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf48 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf282 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___attn_qkv, getattr_l__mod___blocks___2___norm1, x_25, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf43, buf22, buf34, primals_23, primals_29, primals_30, primals_31, buf47, buf48, buf282, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_23
        del primals_29
        del primals_31
        buf49 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf48, reinterpret_tensor(primals_32, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf49)
        del primals_33
        # Source Nodes: [x_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf50 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf49, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf49, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf49, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf51 = buf50[0]
        buf52 = buf50[1]
        buf53 = buf50[2]
        buf54 = buf50[3]
        del buf50
        buf55 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_34, (768, 768), (1, 768), 0), out=buf55)
        buf59 = buf22; del buf22  # reuse
        buf60 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf281 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_37, x_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf43, buf55, primals_35, primals_36, primals_37, buf59, buf60, buf281, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_37
        buf61 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_39, buf60, reinterpret_tensor(primals_38, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf61)
        del primals_39
        buf62 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39, x_42], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf61, buf62, 4866048, grid=grid(4866048), stream=stream0)
        buf63 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf62, reinterpret_tensor(primals_40, (3072, 768), (1, 3072), 0), out=buf63)
        buf64 = reinterpret_tensor(buf63, (8, 198, 768), (152064, 768, 1), 0); del buf63  # reuse
        buf68 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf69 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf280 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___attn_qkv, getattr_l__mod___blocks___3___norm1, x_37, x_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf64, buf43, buf55, primals_35, primals_41, primals_42, primals_43, buf68, buf69, buf280, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_35
        del primals_41
        del primals_43
        buf70 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, buf69, reinterpret_tensor(primals_44, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf70)
        del primals_45
        # Source Nodes: [x_45], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf71 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf70, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf70, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf70, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf72 = buf71[0]
        buf73 = buf71[1]
        buf74 = buf71[2]
        buf75 = buf71[3]
        del buf71
        buf76 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_46, (768, 768), (1, 768), 0), out=buf76)
        buf80 = buf43; del buf43  # reuse
        buf81 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf279 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_49, x_50], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf64, buf76, primals_47, primals_48, primals_49, buf80, buf81, buf279, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_49
        buf82 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_51, buf81, reinterpret_tensor(primals_50, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf82)
        del primals_51
        buf83 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51, x_54], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf82, buf83, 4866048, grid=grid(4866048), stream=stream0)
        buf84 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf83, reinterpret_tensor(primals_52, (3072, 768), (1, 3072), 0), out=buf84)
        buf85 = reinterpret_tensor(buf84, (8, 198, 768), (152064, 768, 1), 0); del buf84  # reuse
        buf89 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf90 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf278 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___attn_qkv, getattr_l__mod___blocks___4___norm1, x_49, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf85, buf64, buf76, primals_47, primals_53, primals_54, primals_55, buf89, buf90, buf278, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_47
        del primals_53
        del primals_55
        buf91 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_57, buf90, reinterpret_tensor(primals_56, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf91)
        del primals_57
        # Source Nodes: [x_57], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf92 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf91, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf91, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf91, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf93 = buf92[0]
        buf94 = buf92[1]
        buf95 = buf92[2]
        buf96 = buf92[3]
        del buf92
        buf97 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), out=buf97)
        buf101 = buf64; del buf64  # reuse
        buf102 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf277 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_61, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf85, buf97, primals_59, primals_60, primals_61, buf101, buf102, buf277, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_61
        buf103 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_63, buf102, reinterpret_tensor(primals_62, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf103)
        del primals_63
        buf104 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_66], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf103, buf104, 4866048, grid=grid(4866048), stream=stream0)
        buf105 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf104, reinterpret_tensor(primals_64, (3072, 768), (1, 3072), 0), out=buf105)
        buf106 = reinterpret_tensor(buf105, (8, 198, 768), (152064, 768, 1), 0); del buf105  # reuse
        buf110 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf111 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf276 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___attn_qkv, getattr_l__mod___blocks___5___norm1, x_61, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf106, buf85, buf97, primals_59, primals_65, primals_66, primals_67, buf110, buf111, buf276, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_59
        del primals_65
        del primals_67
        buf112 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_69, buf111, reinterpret_tensor(primals_68, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf112)
        del primals_69
        # Source Nodes: [x_69], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf113 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf112, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf112, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf112, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf114 = buf113[0]
        buf115 = buf113[1]
        buf116 = buf113[2]
        buf117 = buf113[3]
        del buf113
        buf118 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_70, (768, 768), (1, 768), 0), out=buf118)
        buf122 = buf85; del buf85  # reuse
        buf123 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf275 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_73, x_74], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf106, buf118, primals_71, primals_72, primals_73, buf122, buf123, buf275, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_73
        buf124 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_75, buf123, reinterpret_tensor(primals_74, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf124)
        del primals_75
        buf125 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_75, x_78], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf124, buf125, 4866048, grid=grid(4866048), stream=stream0)
        buf126 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf125, reinterpret_tensor(primals_76, (3072, 768), (1, 3072), 0), out=buf126)
        buf127 = reinterpret_tensor(buf126, (8, 198, 768), (152064, 768, 1), 0); del buf126  # reuse
        buf131 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf132 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf274 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___attn_qkv, getattr_l__mod___blocks___6___norm1, x_73, x_80], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf127, buf106, buf118, primals_71, primals_77, primals_78, primals_79, buf131, buf132, buf274, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_71
        del primals_77
        del primals_79
        buf133 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf132, reinterpret_tensor(primals_80, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf133)
        del primals_81
        # Source Nodes: [x_81], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf134 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf133, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf133, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf133, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf135 = buf134[0]
        buf136 = buf134[1]
        buf137 = buf134[2]
        buf138 = buf134[3]
        del buf134
        buf139 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_82, (768, 768), (1, 768), 0), out=buf139)
        buf143 = buf106; del buf106  # reuse
        buf144 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf273 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_85, x_86], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf127, buf139, primals_83, primals_84, primals_85, buf143, buf144, buf273, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_85
        buf145 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_87, buf144, reinterpret_tensor(primals_86, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf145)
        del primals_87
        buf146 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87, x_90], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf145, buf146, 4866048, grid=grid(4866048), stream=stream0)
        buf147 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf146, reinterpret_tensor(primals_88, (3072, 768), (1, 3072), 0), out=buf147)
        buf148 = reinterpret_tensor(buf147, (8, 198, 768), (152064, 768, 1), 0); del buf147  # reuse
        buf152 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf153 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf272 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___attn_qkv, getattr_l__mod___blocks___7___norm1, x_85, x_92], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf148, buf127, buf139, primals_83, primals_89, primals_90, primals_91, buf152, buf153, buf272, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_83
        del primals_89
        del primals_91
        buf154 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, buf153, reinterpret_tensor(primals_92, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf154)
        del primals_93
        # Source Nodes: [x_93], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf155 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf154, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf154, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf154, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf156 = buf155[0]
        buf157 = buf155[1]
        buf158 = buf155[2]
        buf159 = buf155[3]
        del buf155
        buf160 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_94, (768, 768), (1, 768), 0), out=buf160)
        buf164 = buf127; del buf127  # reuse
        buf165 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf271 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_97, x_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf148, buf160, primals_95, primals_96, primals_97, buf164, buf165, buf271, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_97
        buf166 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_99, buf165, reinterpret_tensor(primals_98, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf166)
        del primals_99
        buf167 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_99], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf166, buf167, 4866048, grid=grid(4866048), stream=stream0)
        buf168 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf167, reinterpret_tensor(primals_100, (3072, 768), (1, 3072), 0), out=buf168)
        buf169 = reinterpret_tensor(buf168, (8, 198, 768), (152064, 768, 1), 0); del buf168  # reuse
        buf173 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf174 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf270 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___attn_qkv, getattr_l__mod___blocks___8___norm1, x_104, x_97], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf169, buf148, buf160, primals_95, primals_101, primals_102, primals_103, buf173, buf174, buf270, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_101
        del primals_103
        del primals_95
        buf175 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_105, buf174, reinterpret_tensor(primals_104, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf175)
        del primals_105
        # Source Nodes: [x_105], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf176 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf175, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf175, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf175, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf177 = buf176[0]
        buf178 = buf176[1]
        buf179 = buf176[2]
        buf180 = buf176[3]
        del buf176
        buf181 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), out=buf181)
        buf185 = buf148; del buf148  # reuse
        buf186 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf269 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_109, x_110], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf169, buf181, primals_107, primals_108, primals_109, buf185, buf186, buf269, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_109
        buf187 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_111, buf186, reinterpret_tensor(primals_110, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf187)
        del primals_111
        buf188 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111, x_114], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf187, buf188, 4866048, grid=grid(4866048), stream=stream0)
        buf189 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf188, reinterpret_tensor(primals_112, (3072, 768), (1, 3072), 0), out=buf189)
        buf190 = reinterpret_tensor(buf189, (8, 198, 768), (152064, 768, 1), 0); del buf189  # reuse
        buf194 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf195 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf268 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___attn_qkv, getattr_l__mod___blocks___9___norm1, x_109, x_116], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf190, buf169, buf181, primals_107, primals_113, primals_114, primals_115, buf194, buf195, buf268, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_107
        del primals_113
        del primals_115
        buf196 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_117, buf195, reinterpret_tensor(primals_116, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf196)
        del primals_117
        # Source Nodes: [x_117], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf197 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf198 = buf197[0]
        buf199 = buf197[1]
        buf200 = buf197[2]
        buf201 = buf197[3]
        del buf197
        buf202 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), out=buf202)
        buf206 = buf169; del buf169  # reuse
        buf207 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf267 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_121, x_122], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf190, buf202, primals_119, primals_120, primals_121, buf206, buf207, buf267, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_121
        buf208 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_123, buf207, reinterpret_tensor(primals_122, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf208)
        del primals_123
        buf209 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123, x_126], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf208, buf209, 4866048, grid=grid(4866048), stream=stream0)
        buf210 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf209, reinterpret_tensor(primals_124, (3072, 768), (1, 3072), 0), out=buf210)
        buf211 = reinterpret_tensor(buf210, (8, 198, 768), (152064, 768, 1), 0); del buf210  # reuse
        buf215 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf216 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf266 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___attn_qkv, getattr_l__mod___blocks___10___norm1, x_121, x_128], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf211, buf190, buf202, primals_119, primals_125, primals_126, primals_127, buf215, buf216, buf266, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_119
        del primals_125
        del primals_127
        buf217 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf216, reinterpret_tensor(primals_128, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf217)
        del primals_129
        # Source Nodes: [x_129], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf218 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf217, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf217, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf217, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf219 = buf218[0]
        buf220 = buf218[1]
        buf221 = buf218[2]
        buf222 = buf218[3]
        del buf218
        buf223 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_130, (768, 768), (1, 768), 0), out=buf223)
        buf227 = buf190; del buf190  # reuse
        buf228 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf265 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_133, x_134], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf211, buf223, primals_131, primals_132, primals_133, buf227, buf228, buf265, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_133
        buf229 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_135, buf228, reinterpret_tensor(primals_134, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf229)
        del primals_135
        buf230 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135, x_138], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf229, buf230, 4866048, grid=grid(4866048), stream=stream0)
        buf231 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf230, reinterpret_tensor(primals_136, (3072, 768), (1, 3072), 0), out=buf231)
        buf232 = reinterpret_tensor(buf231, (8, 198, 768), (152064, 768, 1), 0); del buf231  # reuse
        buf236 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf237 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf264 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___attn_qkv, getattr_l__mod___blocks___11___norm1, x_133, x_140], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf232, buf211, buf223, primals_131, primals_137, primals_138, primals_139, buf236, buf237, buf264, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_131
        del primals_137
        del primals_139
        buf238 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_141, buf237, reinterpret_tensor(primals_140, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf238)
        del primals_141
        # Source Nodes: [x_141], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf239 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf238, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf238, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf238, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, True)
        buf240 = buf239[0]
        buf241 = buf239[1]
        buf242 = buf239[2]
        buf243 = buf239[3]
        del buf239
        buf244 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_142, (768, 768), (1, 768), 0), out=buf244)
        buf248 = buf211; del buf211  # reuse
        buf249 = empty((1584, 768), device='cuda', dtype=torch.float32)
        buf263 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_145, x_146], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_1.run(buf232, buf244, primals_143, primals_144, primals_145, buf248, buf249, buf263, 1584, 768, grid=grid(1584), stream=stream0)
        del primals_145
        buf250 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_147, buf249, reinterpret_tensor(primals_146, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf250)
        del primals_147
        buf251 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147, x_150], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_2.run(buf250, buf251, 4866048, grid=grid(4866048), stream=stream0)
        buf252 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf251, reinterpret_tensor(primals_148, (3072, 768), (1, 3072), 0), out=buf252)
        buf253 = reinterpret_tensor(buf252, (8, 198, 768), (152064, 768, 1), 0); del buf252  # reuse
        buf257 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf258 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf262 = empty((8, 198, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145, x_153, x_155], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf253, buf232, buf244, primals_143, primals_149, primals_150, primals_151, buf257, buf258, buf262, 1584, 768, grid=grid(1584), stream=stream0)
        del buf232
        del buf244
        del buf253
        del primals_143
        del primals_149
        del primals_151
        buf259 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (8, 768), (152064, 1), 0), reinterpret_tensor(primals_152, (768, 1000), (1, 768), 0), out=buf259)
        buf260 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (8, 768), (152064, 1), 768), reinterpret_tensor(primals_154, (768, 1000), (1, 768), 0), out=buf260)
        buf261 = buf259; del buf259  # reuse
        # Source Nodes: [add_25, pred], Original ATen: [aten.add, aten.div]
        triton_poi_fused_add_div_4.run(buf261, primals_153, buf260, primals_155, 8000, grid=grid(8000), stream=stream0)
        del buf260
        del primals_153
        del primals_155
        return (buf261, primals_4, primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, primals_156, buf5, buf6, reinterpret_tensor(buf7, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf7, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf7, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf10, buf11, buf12, reinterpret_tensor(buf9, (1584, 768), (768, 1), 0), buf17, buf18, buf19, buf20, buf26, buf27, reinterpret_tensor(buf28, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf28, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf28, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf31, buf32, buf33, reinterpret_tensor(buf30, (1584, 768), (768, 1), 0), buf38, buf39, buf40, buf41, buf47, buf48, reinterpret_tensor(buf49, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf49, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf49, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf52, buf53, buf54, reinterpret_tensor(buf51, (1584, 768), (768, 1), 0), buf59, buf60, buf61, buf62, buf68, buf69, reinterpret_tensor(buf70, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf70, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf70, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf73, buf74, buf75, reinterpret_tensor(buf72, (1584, 768), (768, 1), 0), buf80, buf81, buf82, buf83, buf89, buf90, reinterpret_tensor(buf91, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf91, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf91, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf94, buf95, buf96, reinterpret_tensor(buf93, (1584, 768), (768, 1), 0), buf101, buf102, buf103, buf104, buf110, buf111, reinterpret_tensor(buf112, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf112, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf112, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf115, buf116, buf117, reinterpret_tensor(buf114, (1584, 768), (768, 1), 0), buf122, buf123, buf124, buf125, buf131, buf132, reinterpret_tensor(buf133, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf133, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf133, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf136, buf137, buf138, reinterpret_tensor(buf135, (1584, 768), (768, 1), 0), buf143, buf144, buf145, buf146, buf152, buf153, reinterpret_tensor(buf154, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf154, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf154, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf157, buf158, buf159, reinterpret_tensor(buf156, (1584, 768), (768, 1), 0), buf164, buf165, buf166, buf167, buf173, buf174, reinterpret_tensor(buf175, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf175, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf175, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf178, buf179, buf180, reinterpret_tensor(buf177, (1584, 768), (768, 1), 0), buf185, buf186, buf187, buf188, buf194, buf195, reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf199, buf200, buf201, reinterpret_tensor(buf198, (1584, 768), (768, 1), 0), buf206, buf207, buf208, buf209, buf215, buf216, reinterpret_tensor(buf217, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf217, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf217, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf220, buf221, buf222, reinterpret_tensor(buf219, (1584, 768), (768, 1), 0), buf227, buf228, buf229, buf230, buf236, buf237, reinterpret_tensor(buf238, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf238, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf238, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf241, buf242, buf243, reinterpret_tensor(buf240, (1584, 768), (768, 1), 0), buf248, buf249, buf250, buf251, buf257, reinterpret_tensor(buf258, (8, 768), (152064, 1), 0), reinterpret_tensor(buf258, (8, 768), (152064, 1), 768), reinterpret_tensor(primals_154, (1000, 768), (768, 1), 0), reinterpret_tensor(primals_152, (1000, 768), (768, 1), 0), buf262, reinterpret_tensor(primals_148, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_146, (3072, 768), (768, 1), 0), buf263, reinterpret_tensor(primals_142, (768, 768), (768, 1), 0), buf240, reinterpret_tensor(primals_140, (2304, 768), (768, 1), 0), buf264, reinterpret_tensor(primals_136, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_134, (3072, 768), (768, 1), 0), buf265, reinterpret_tensor(primals_130, (768, 768), (768, 1), 0), buf219, reinterpret_tensor(primals_128, (2304, 768), (768, 1), 0), buf266, reinterpret_tensor(primals_124, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_122, (3072, 768), (768, 1), 0), buf267, reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), buf198, reinterpret_tensor(primals_116, (2304, 768), (768, 1), 0), buf268, reinterpret_tensor(primals_112, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_110, (3072, 768), (768, 1), 0), buf269, reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), buf177, reinterpret_tensor(primals_104, (2304, 768), (768, 1), 0), buf270, reinterpret_tensor(primals_100, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_98, (3072, 768), (768, 1), 0), buf271, reinterpret_tensor(primals_94, (768, 768), (768, 1), 0), buf156, reinterpret_tensor(primals_92, (2304, 768), (768, 1), 0), buf272, reinterpret_tensor(primals_88, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_86, (3072, 768), (768, 1), 0), buf273, reinterpret_tensor(primals_82, (768, 768), (768, 1), 0), buf135, reinterpret_tensor(primals_80, (2304, 768), (768, 1), 0), buf274, reinterpret_tensor(primals_76, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_74, (3072, 768), (768, 1), 0), buf275, reinterpret_tensor(primals_70, (768, 768), (768, 1), 0), buf114, reinterpret_tensor(primals_68, (2304, 768), (768, 1), 0), buf276, reinterpret_tensor(primals_64, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_62, (3072, 768), (768, 1), 0), buf277, reinterpret_tensor(primals_58, (768, 768), (768, 1), 0), buf93, reinterpret_tensor(primals_56, (2304, 768), (768, 1), 0), buf278, reinterpret_tensor(primals_52, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_50, (3072, 768), (768, 1), 0), buf279, reinterpret_tensor(primals_46, (768, 768), (768, 1), 0), buf72, reinterpret_tensor(primals_44, (2304, 768), (768, 1), 0), buf280, reinterpret_tensor(primals_40, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_38, (3072, 768), (768, 1), 0), buf281, reinterpret_tensor(primals_34, (768, 768), (768, 1), 0), buf51, reinterpret_tensor(primals_32, (2304, 768), (768, 1), 0), buf282, reinterpret_tensor(primals_28, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_26, (3072, 768), (768, 1), 0), buf283, reinterpret_tensor(primals_22, (768, 768), (768, 1), 0), buf30, reinterpret_tensor(primals_20, (2304, 768), (768, 1), 0), buf284, reinterpret_tensor(primals_16, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_14, (3072, 768), (768, 1), 0), buf285, reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), buf9, reinterpret_tensor(primals_8, (2304, 768), (768, 1), 0), buf286, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('deit_base_distilled_patch16_224', benchmark_compiled_module)
