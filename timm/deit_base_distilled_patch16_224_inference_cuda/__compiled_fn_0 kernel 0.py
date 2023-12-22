
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


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gco4csik6z2rdlrwnkwijyoufitolgcl6m7q3rxzb5k7rpgdr3.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# getattr_l__mod___blocks___0___norm1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# x_5 => add
triton_per_fused_add_cat_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp53, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jd6y7zxszzahlqx65jzrrqcyc5dbkgifyl5u5daarkmmshdw5c.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_13], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# x_13 => add_3
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bh/cbhzbmdnio4oed4fommcrr6v7tg4ofnjlsmf6ya6yvsb4v6ovjte.py
# Source Nodes: [x_15], Original ATen: [aten.gelu]
# x_15 => add_6, erf, mul_4, mul_5, mul_6
triton_poi_fused_gelu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4866048
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


# kernel path: /tmp/torchinductor_youkaichao/jx/cjxr3e7dq2k36yubujfq2d3fdxac6olgapfymomwkabdnhrhcu6b.py
# Source Nodes: [getattr_l__mod___blocks___1___norm1, x_13, x_20], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm1 => add_8, add_9, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
# x_13 => add_3
# x_20 => add_7
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwnjn5dz45ekxecf5xvcgwi5ua6imwpilz5svyd2skh4ystnzcx.py
# Source Nodes: [add_25, x_158], Original ATen: [aten.add, aten.div]
# add_25 => add_87
# x_158 => div
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 198, 768), (152064, 768, 1))
    assert_size_stride(arg1_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg2_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg3_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (2304, 768), (768, 1))
    assert_size_stride(arg8_1, (2304, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (3072, 768), (768, 1))
    assert_size_stride(arg14_1, (3072, ), (1, ))
    assert_size_stride(arg15_1, (768, 3072), (3072, 1))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (2304, 768), (768, 1))
    assert_size_stride(arg20_1, (2304, ), (1, ))
    assert_size_stride(arg21_1, (768, 768), (768, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (3072, 768), (768, 1))
    assert_size_stride(arg26_1, (3072, ), (1, ))
    assert_size_stride(arg27_1, (768, 3072), (3072, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (2304, 768), (768, 1))
    assert_size_stride(arg32_1, (2304, ), (1, ))
    assert_size_stride(arg33_1, (768, 768), (768, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (3072, 768), (768, 1))
    assert_size_stride(arg38_1, (3072, ), (1, ))
    assert_size_stride(arg39_1, (768, 3072), (3072, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (2304, 768), (768, 1))
    assert_size_stride(arg44_1, (2304, ), (1, ))
    assert_size_stride(arg45_1, (768, 768), (768, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (3072, 768), (768, 1))
    assert_size_stride(arg50_1, (3072, ), (1, ))
    assert_size_stride(arg51_1, (768, 3072), (3072, 1))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (2304, 768), (768, 1))
    assert_size_stride(arg56_1, (2304, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (3072, 768), (768, 1))
    assert_size_stride(arg62_1, (3072, ), (1, ))
    assert_size_stride(arg63_1, (768, 3072), (3072, 1))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (2304, 768), (768, 1))
    assert_size_stride(arg68_1, (2304, ), (1, ))
    assert_size_stride(arg69_1, (768, 768), (768, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (3072, 768), (768, 1))
    assert_size_stride(arg74_1, (3072, ), (1, ))
    assert_size_stride(arg75_1, (768, 3072), (3072, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (2304, 768), (768, 1))
    assert_size_stride(arg80_1, (2304, ), (1, ))
    assert_size_stride(arg81_1, (768, 768), (768, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (3072, 768), (768, 1))
    assert_size_stride(arg86_1, (3072, ), (1, ))
    assert_size_stride(arg87_1, (768, 3072), (3072, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (2304, 768), (768, 1))
    assert_size_stride(arg92_1, (2304, ), (1, ))
    assert_size_stride(arg93_1, (768, 768), (768, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (3072, 768), (768, 1))
    assert_size_stride(arg98_1, (3072, ), (1, ))
    assert_size_stride(arg99_1, (768, 3072), (3072, 1))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (2304, 768), (768, 1))
    assert_size_stride(arg104_1, (2304, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (3072, 768), (768, 1))
    assert_size_stride(arg110_1, (3072, ), (1, ))
    assert_size_stride(arg111_1, (768, 3072), (3072, 1))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (2304, 768), (768, 1))
    assert_size_stride(arg116_1, (2304, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (3072, 768), (768, 1))
    assert_size_stride(arg122_1, (3072, ), (1, ))
    assert_size_stride(arg123_1, (768, 3072), (3072, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (2304, 768), (768, 1))
    assert_size_stride(arg128_1, (2304, ), (1, ))
    assert_size_stride(arg129_1, (768, 768), (768, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (3072, 768), (768, 1))
    assert_size_stride(arg134_1, (3072, ), (1, ))
    assert_size_stride(arg135_1, (768, 3072), (3072, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (2304, 768), (768, 1))
    assert_size_stride(arg140_1, (2304, ), (1, ))
    assert_size_stride(arg141_1, (768, 768), (768, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (3072, 768), (768, 1))
    assert_size_stride(arg146_1, (3072, ), (1, ))
    assert_size_stride(arg147_1, (768, 3072), (3072, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (1000, 768), (768, 1))
    assert_size_stride(arg152_1, (1000, ), (1, ))
    assert_size_stride(arg153_1, (1000, 768), (768, 1))
    assert_size_stride(arg154_1, (1000, ), (1, ))
    assert_size_stride(arg155_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg155_1, arg3_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg155_1
        del arg3_1
        buf1 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_cat_native_layer_norm_0.run(arg1_1, arg2_1, buf0, arg4_1, arg0_1, arg5_1, arg6_1, buf1, buf5, 1584, 768, grid=grid(1584), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg4_1
        del arg5_1
        del arg6_1
        del buf0
        buf6 = empty((1584, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf5, (1584, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del arg7_1
        del arg8_1
        # Source Nodes: [x_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf7 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf6, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf6, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf6, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf8 = buf7[0]
        del buf7
        buf12 = reinterpret_tensor(buf5, (1584, 768), (768, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf8, (1584, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), out=buf12)
        del arg9_1
        buf16 = reinterpret_tensor(buf8, (8, 198, 768), (152064, 768, 1), 0); del buf8  # reuse
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf1, buf12, arg10_1, arg11_1, arg12_1, buf16, 1584, 768, grid=grid(1584), stream=stream0)
        del arg11_1
        del arg12_1
        buf17 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf16, (1584, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 3072), (1, 768), 0), out=buf17)
        del arg13_1
        buf18 = reinterpret_tensor(buf17, (8, 198, 3072), (608256, 3072, 1), 0); del buf17  # reuse
        # Source Nodes: [x_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf18, arg14_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg14_1
        buf19 = reinterpret_tensor(buf16, (1584, 768), (768, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg15_1, (3072, 768), (1, 3072), 0), out=buf19)
        del arg15_1
        buf20 = reinterpret_tensor(buf19, (8, 198, 768), (152064, 768, 1), 0); del buf19  # reuse
        buf24 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm1, x_13, x_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf20, buf1, buf12, arg10_1, arg16_1, arg17_1, arg18_1, buf24, 1584, 768, grid=grid(1584), stream=stream0)
        del arg10_1
        del arg16_1
        del arg17_1
        del arg18_1
        del buf1
        buf25 = buf6; del buf6  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg20_1, reinterpret_tensor(buf24, (1584, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del arg19_1
        del arg20_1
        # Source Nodes: [x_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf26 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf25, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf25, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf25, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf27 = buf26[0]
        del buf26
        buf31 = reinterpret_tensor(buf24, (1584, 768), (768, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1584, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), out=buf31)
        del arg21_1
        buf35 = reinterpret_tensor(buf27, (8, 198, 768), (152064, 768, 1), 0); del buf27  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf20, buf31, arg22_1, arg23_1, arg24_1, buf35, 1584, 768, grid=grid(1584), stream=stream0)
        del arg23_1
        del arg24_1
        buf36 = reinterpret_tensor(buf18, (1584, 3072), (3072, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (1584, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 3072), (1, 768), 0), out=buf36)
        del arg25_1
        buf37 = reinterpret_tensor(buf36, (8, 198, 3072), (608256, 3072, 1), 0); del buf36  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf37, arg26_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg26_1
        buf38 = reinterpret_tensor(buf35, (1584, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg27_1, (3072, 768), (1, 3072), 0), out=buf38)
        del arg27_1
        buf39 = reinterpret_tensor(buf38, (8, 198, 768), (152064, 768, 1), 0); del buf38  # reuse
        buf43 = reinterpret_tensor(buf12, (8, 198, 768), (152064, 768, 1), 0); del buf12  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm1, x_25, x_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf39, buf20, buf31, arg22_1, arg28_1, arg29_1, arg30_1, buf43, 1584, 768, grid=grid(1584), stream=stream0)
        del arg22_1
        del arg28_1
        del arg29_1
        del arg30_1
        del buf20
        buf44 = buf25; del buf25  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg32_1, reinterpret_tensor(buf43, (1584, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf44)
        del arg31_1
        del arg32_1
        # Source Nodes: [x_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf45 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf44, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf44, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf44, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf46 = buf45[0]
        del buf45
        buf50 = reinterpret_tensor(buf43, (1584, 768), (768, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (1584, 768), (768, 1), 0), reinterpret_tensor(arg33_1, (768, 768), (1, 768), 0), out=buf50)
        del arg33_1
        buf54 = reinterpret_tensor(buf46, (8, 198, 768), (152064, 768, 1), 0); del buf46  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf39, buf50, arg34_1, arg35_1, arg36_1, buf54, 1584, 768, grid=grid(1584), stream=stream0)
        del arg35_1
        del arg36_1
        buf55 = reinterpret_tensor(buf37, (1584, 3072), (3072, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (1584, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 3072), (1, 768), 0), out=buf55)
        del arg37_1
        buf56 = reinterpret_tensor(buf55, (8, 198, 3072), (608256, 3072, 1), 0); del buf55  # reuse
        # Source Nodes: [x_39], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf56, arg38_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg38_1
        buf57 = reinterpret_tensor(buf54, (1584, 768), (768, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg39_1, (3072, 768), (1, 3072), 0), out=buf57)
        del arg39_1
        buf58 = reinterpret_tensor(buf57, (8, 198, 768), (152064, 768, 1), 0); del buf57  # reuse
        buf62 = reinterpret_tensor(buf31, (8, 198, 768), (152064, 768, 1), 0); del buf31  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm1, x_37, x_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf58, buf39, buf50, arg34_1, arg40_1, arg41_1, arg42_1, buf62, 1584, 768, grid=grid(1584), stream=stream0)
        del arg34_1
        del arg40_1
        del arg41_1
        del arg42_1
        del buf39
        buf63 = buf44; del buf44  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg44_1, reinterpret_tensor(buf62, (1584, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf63)
        del arg43_1
        del arg44_1
        # Source Nodes: [x_45], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf64 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf63, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf63, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf63, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf65 = buf64[0]
        del buf64
        buf69 = reinterpret_tensor(buf62, (1584, 768), (768, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (1584, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 768), (1, 768), 0), out=buf69)
        del arg45_1
        buf73 = reinterpret_tensor(buf65, (8, 198, 768), (152064, 768, 1), 0); del buf65  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf58, buf69, arg46_1, arg47_1, arg48_1, buf73, 1584, 768, grid=grid(1584), stream=stream0)
        del arg47_1
        del arg48_1
        buf74 = reinterpret_tensor(buf56, (1584, 3072), (3072, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (1584, 768), (768, 1), 0), reinterpret_tensor(arg49_1, (768, 3072), (1, 768), 0), out=buf74)
        del arg49_1
        buf75 = reinterpret_tensor(buf74, (8, 198, 3072), (608256, 3072, 1), 0); del buf74  # reuse
        # Source Nodes: [x_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf75, arg50_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg50_1
        buf76 = reinterpret_tensor(buf73, (1584, 768), (768, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg51_1, (3072, 768), (1, 3072), 0), out=buf76)
        del arg51_1
        buf77 = reinterpret_tensor(buf76, (8, 198, 768), (152064, 768, 1), 0); del buf76  # reuse
        buf81 = reinterpret_tensor(buf50, (8, 198, 768), (152064, 768, 1), 0); del buf50  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm1, x_49, x_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf77, buf58, buf69, arg46_1, arg52_1, arg53_1, arg54_1, buf81, 1584, 768, grid=grid(1584), stream=stream0)
        del arg46_1
        del arg52_1
        del arg53_1
        del arg54_1
        del buf58
        buf82 = buf63; del buf63  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg56_1, reinterpret_tensor(buf81, (1584, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf82)
        del arg55_1
        del arg56_1
        # Source Nodes: [x_57], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf83 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf82, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf82, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf82, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf84 = buf83[0]
        del buf83
        buf88 = reinterpret_tensor(buf81, (1584, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (1584, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), out=buf88)
        del arg57_1
        buf92 = reinterpret_tensor(buf84, (8, 198, 768), (152064, 768, 1), 0); del buf84  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_61], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf77, buf88, arg58_1, arg59_1, arg60_1, buf92, 1584, 768, grid=grid(1584), stream=stream0)
        del arg59_1
        del arg60_1
        buf93 = reinterpret_tensor(buf75, (1584, 3072), (3072, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1584, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 3072), (1, 768), 0), out=buf93)
        del arg61_1
        buf94 = reinterpret_tensor(buf93, (8, 198, 3072), (608256, 3072, 1), 0); del buf93  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf94, arg62_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg62_1
        buf95 = reinterpret_tensor(buf92, (1584, 768), (768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg63_1, (3072, 768), (1, 3072), 0), out=buf95)
        del arg63_1
        buf96 = reinterpret_tensor(buf95, (8, 198, 768), (152064, 768, 1), 0); del buf95  # reuse
        buf100 = reinterpret_tensor(buf69, (8, 198, 768), (152064, 768, 1), 0); del buf69  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm1, x_61, x_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf96, buf77, buf88, arg58_1, arg64_1, arg65_1, arg66_1, buf100, 1584, 768, grid=grid(1584), stream=stream0)
        del arg58_1
        del arg64_1
        del arg65_1
        del arg66_1
        del buf77
        buf101 = buf82; del buf82  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg68_1, reinterpret_tensor(buf100, (1584, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf101)
        del arg67_1
        del arg68_1
        # Source Nodes: [x_69], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf102 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf101, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf101, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf101, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf103 = buf102[0]
        del buf102
        buf107 = reinterpret_tensor(buf100, (1584, 768), (768, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (1584, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), out=buf107)
        del arg69_1
        buf111 = reinterpret_tensor(buf103, (8, 198, 768), (152064, 768, 1), 0); del buf103  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_73], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf96, buf107, arg70_1, arg71_1, arg72_1, buf111, 1584, 768, grid=grid(1584), stream=stream0)
        del arg71_1
        del arg72_1
        buf112 = reinterpret_tensor(buf94, (1584, 3072), (3072, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (1584, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 3072), (1, 768), 0), out=buf112)
        del arg73_1
        buf113 = reinterpret_tensor(buf112, (8, 198, 3072), (608256, 3072, 1), 0); del buf112  # reuse
        # Source Nodes: [x_75], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf113, arg74_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg74_1
        buf114 = reinterpret_tensor(buf111, (1584, 768), (768, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg75_1, (3072, 768), (1, 3072), 0), out=buf114)
        del arg75_1
        buf115 = reinterpret_tensor(buf114, (8, 198, 768), (152064, 768, 1), 0); del buf114  # reuse
        buf119 = reinterpret_tensor(buf88, (8, 198, 768), (152064, 768, 1), 0); del buf88  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm1, x_73, x_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf115, buf96, buf107, arg70_1, arg76_1, arg77_1, arg78_1, buf119, 1584, 768, grid=grid(1584), stream=stream0)
        del arg70_1
        del arg76_1
        del arg77_1
        del arg78_1
        del buf107
        buf120 = buf101; del buf101  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg80_1, reinterpret_tensor(buf119, (1584, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf120)
        del arg79_1
        del arg80_1
        # Source Nodes: [x_81], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf121 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf120, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf120, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf120, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf122 = buf121[0]
        del buf121
        buf126 = reinterpret_tensor(buf119, (1584, 768), (768, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1584, 768), (768, 1), 0), reinterpret_tensor(arg81_1, (768, 768), (1, 768), 0), out=buf126)
        del arg81_1
        buf130 = reinterpret_tensor(buf122, (8, 198, 768), (152064, 768, 1), 0); del buf122  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_85], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf115, buf126, arg82_1, arg83_1, arg84_1, buf130, 1584, 768, grid=grid(1584), stream=stream0)
        del arg83_1
        del arg84_1
        buf131 = reinterpret_tensor(buf113, (1584, 3072), (3072, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (1584, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 3072), (1, 768), 0), out=buf131)
        del arg85_1
        buf132 = reinterpret_tensor(buf131, (8, 198, 3072), (608256, 3072, 1), 0); del buf131  # reuse
        # Source Nodes: [x_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf132, arg86_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg86_1
        buf133 = reinterpret_tensor(buf130, (1584, 768), (768, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg87_1, (3072, 768), (1, 3072), 0), out=buf133)
        del arg87_1
        buf134 = reinterpret_tensor(buf133, (8, 198, 768), (152064, 768, 1), 0); del buf133  # reuse
        buf138 = buf96; del buf96  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm1, x_85, x_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf134, buf115, buf126, arg82_1, arg88_1, arg89_1, arg90_1, buf138, 1584, 768, grid=grid(1584), stream=stream0)
        del arg82_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf115
        buf139 = buf120; del buf120  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg92_1, reinterpret_tensor(buf138, (1584, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf139)
        del arg91_1
        del arg92_1
        # Source Nodes: [x_93], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf140 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf139, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf139, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf139, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf141 = buf140[0]
        del buf140
        buf145 = reinterpret_tensor(buf138, (1584, 768), (768, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (1584, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 768), (1, 768), 0), out=buf145)
        del arg93_1
        buf149 = reinterpret_tensor(buf141, (8, 198, 768), (152064, 768, 1), 0); del buf141  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_97], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf134, buf145, arg94_1, arg95_1, arg96_1, buf149, 1584, 768, grid=grid(1584), stream=stream0)
        del arg95_1
        del arg96_1
        buf150 = reinterpret_tensor(buf132, (1584, 3072), (3072, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (1584, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 3072), (1, 768), 0), out=buf150)
        del arg97_1
        buf151 = reinterpret_tensor(buf150, (8, 198, 3072), (608256, 3072, 1), 0); del buf150  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf151, arg98_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg98_1
        buf152 = reinterpret_tensor(buf149, (1584, 768), (768, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg99_1, (3072, 768), (1, 3072), 0), out=buf152)
        del arg99_1
        buf153 = reinterpret_tensor(buf152, (8, 198, 768), (152064, 768, 1), 0); del buf152  # reuse
        buf157 = reinterpret_tensor(buf126, (8, 198, 768), (152064, 768, 1), 0); del buf126  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm1, x_104, x_97], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf153, buf134, buf145, arg94_1, arg100_1, arg101_1, arg102_1, buf157, 1584, 768, grid=grid(1584), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg94_1
        del buf134
        buf158 = buf139; del buf139  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg104_1, reinterpret_tensor(buf157, (1584, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf158)
        del arg103_1
        del arg104_1
        # Source Nodes: [x_105], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf159 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf158, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf158, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf158, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf160 = buf159[0]
        del buf159
        buf164 = reinterpret_tensor(buf157, (1584, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (1584, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), out=buf164)
        del arg105_1
        buf168 = reinterpret_tensor(buf160, (8, 198, 768), (152064, 768, 1), 0); del buf160  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_109], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf153, buf164, arg106_1, arg107_1, arg108_1, buf168, 1584, 768, grid=grid(1584), stream=stream0)
        del arg107_1
        del arg108_1
        buf169 = reinterpret_tensor(buf151, (1584, 3072), (3072, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (1584, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 3072), (1, 768), 0), out=buf169)
        del arg109_1
        buf170 = reinterpret_tensor(buf169, (8, 198, 3072), (608256, 3072, 1), 0); del buf169  # reuse
        # Source Nodes: [x_111], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf170, arg110_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg110_1
        buf171 = reinterpret_tensor(buf168, (1584, 768), (768, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg111_1, (3072, 768), (1, 3072), 0), out=buf171)
        del arg111_1
        buf172 = reinterpret_tensor(buf171, (8, 198, 768), (152064, 768, 1), 0); del buf171  # reuse
        buf176 = reinterpret_tensor(buf145, (8, 198, 768), (152064, 768, 1), 0); del buf145  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm1, x_109, x_116], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf172, buf153, buf164, arg106_1, arg112_1, arg113_1, arg114_1, buf176, 1584, 768, grid=grid(1584), stream=stream0)
        del arg106_1
        del arg112_1
        del arg113_1
        del arg114_1
        del buf153
        buf177 = buf158; del buf158  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg116_1, reinterpret_tensor(buf176, (1584, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf177)
        del arg115_1
        del arg116_1
        # Source Nodes: [x_117], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf178 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf177, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf177, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf177, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf179 = buf178[0]
        del buf178
        buf183 = reinterpret_tensor(buf176, (1584, 768), (768, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (1584, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf183)
        del arg117_1
        buf187 = reinterpret_tensor(buf179, (8, 198, 768), (152064, 768, 1), 0); del buf179  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_121], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf172, buf183, arg118_1, arg119_1, arg120_1, buf187, 1584, 768, grid=grid(1584), stream=stream0)
        del arg119_1
        del arg120_1
        buf188 = reinterpret_tensor(buf170, (1584, 3072), (3072, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1584, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 3072), (1, 768), 0), out=buf188)
        del arg121_1
        buf189 = reinterpret_tensor(buf188, (8, 198, 3072), (608256, 3072, 1), 0); del buf188  # reuse
        # Source Nodes: [x_123], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf189, arg122_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg122_1
        buf190 = reinterpret_tensor(buf187, (1584, 768), (768, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg123_1, (3072, 768), (1, 3072), 0), out=buf190)
        del arg123_1
        buf191 = reinterpret_tensor(buf190, (8, 198, 768), (152064, 768, 1), 0); del buf190  # reuse
        buf195 = reinterpret_tensor(buf164, (8, 198, 768), (152064, 768, 1), 0); del buf164  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm1, x_121, x_128], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf191, buf172, buf183, arg118_1, arg124_1, arg125_1, arg126_1, buf195, 1584, 768, grid=grid(1584), stream=stream0)
        del arg118_1
        del arg124_1
        del arg125_1
        del arg126_1
        del buf172
        buf196 = buf177; del buf177  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg128_1, reinterpret_tensor(buf195, (1584, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf196)
        del arg127_1
        del arg128_1
        # Source Nodes: [x_129], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf197 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf198 = buf197[0]
        del buf197
        buf202 = reinterpret_tensor(buf195, (1584, 768), (768, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (1584, 768), (768, 1), 0), reinterpret_tensor(arg129_1, (768, 768), (1, 768), 0), out=buf202)
        del arg129_1
        buf206 = reinterpret_tensor(buf198, (8, 198, 768), (152064, 768, 1), 0); del buf198  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_133], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf191, buf202, arg130_1, arg131_1, arg132_1, buf206, 1584, 768, grid=grid(1584), stream=stream0)
        del arg131_1
        del arg132_1
        buf207 = reinterpret_tensor(buf189, (1584, 3072), (3072, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1584, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 3072), (1, 768), 0), out=buf207)
        del arg133_1
        buf208 = reinterpret_tensor(buf207, (8, 198, 3072), (608256, 3072, 1), 0); del buf207  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf208, arg134_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg134_1
        buf209 = reinterpret_tensor(buf206, (1584, 768), (768, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg135_1, (3072, 768), (1, 3072), 0), out=buf209)
        del arg135_1
        buf210 = reinterpret_tensor(buf209, (8, 198, 768), (152064, 768, 1), 0); del buf209  # reuse
        buf214 = reinterpret_tensor(buf183, (8, 198, 768), (152064, 768, 1), 0); del buf183  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm1, x_133, x_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf210, buf191, buf202, arg130_1, arg136_1, arg137_1, arg138_1, buf214, 1584, 768, grid=grid(1584), stream=stream0)
        del arg130_1
        del arg136_1
        del arg137_1
        del arg138_1
        del buf191
        buf215 = buf196; del buf196  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg140_1, reinterpret_tensor(buf214, (1584, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf215)
        del arg139_1
        del arg140_1
        # Source Nodes: [x_141], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf216 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf215, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf215, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf215, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        del buf215
        buf217 = buf216[0]
        del buf216
        buf221 = reinterpret_tensor(buf214, (1584, 768), (768, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (1584, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), out=buf221)
        del arg141_1
        buf225 = reinterpret_tensor(buf217, (8, 198, 768), (152064, 768, 1), 0); del buf217  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_145], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf210, buf221, arg142_1, arg143_1, arg144_1, buf225, 1584, 768, grid=grid(1584), stream=stream0)
        del arg143_1
        del arg144_1
        buf226 = reinterpret_tensor(buf208, (1584, 3072), (3072, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf225, (1584, 768), (768, 1), 0), reinterpret_tensor(arg145_1, (768, 3072), (1, 768), 0), out=buf226)
        del arg145_1
        buf227 = reinterpret_tensor(buf226, (8, 198, 3072), (608256, 3072, 1), 0); del buf226  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf227, arg146_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg146_1
        buf228 = reinterpret_tensor(buf225, (1584, 768), (768, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg147_1, (3072, 768), (1, 3072), 0), out=buf228)
        del arg147_1
        del buf227
        buf229 = reinterpret_tensor(buf228, (8, 198, 768), (152064, 768, 1), 0); del buf228  # reuse
        buf233 = reinterpret_tensor(buf202, (8, 198, 768), (152064, 768, 1), 0); del buf202  # reuse
        # Source Nodes: [x_145, x_153, x_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf229, buf210, buf221, arg142_1, arg148_1, arg149_1, arg150_1, buf233, 1584, 768, grid=grid(1584), stream=stream0)
        del arg142_1
        del arg148_1
        del arg149_1
        del arg150_1
        del buf210
        del buf221
        del buf229
        buf234 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (8, 768), (152064, 1), 0), reinterpret_tensor(arg151_1, (768, 1000), (1, 768), 0), out=buf234)
        del arg151_1
        buf235 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (8, 768), (152064, 1), 768), reinterpret_tensor(arg153_1, (768, 1000), (1, 768), 0), out=buf235)
        del arg153_1
        del buf233
        buf236 = buf234; del buf234  # reuse
        # Source Nodes: [add_25, x_158], Original ATen: [aten.add, aten.div]
        triton_poi_fused_add_div_4.run(buf236, arg152_1, buf235, arg154_1, 8000, grid=grid(8000), stream=stream0)
        del arg152_1
        del arg154_1
        return (buf236, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('deit_base_distilled_patch16_224', benchmark_compiled_module)
