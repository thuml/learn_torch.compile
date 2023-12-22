
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


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrrn7c2lsrw45hri36vlccrnbmjg5nf6og3jxxezw7o2wr5qcfd.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# inputs_embeds => embedding
# position_embeddings => embedding_2
# token_type_embeddings => embedding_1
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 30522
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 30522), "index out of bounds: 0 <= tmp3 < 30522")
    tmp4 = tl.load(in_ptr1 + (r2 + (768*tmp3)), rmask, other=0.0)
    tmp6 = tmp5 + 2
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert((0 <= tmp8) & (tmp8 < 2), "index out of bounds: 0 <= tmp8 < 2")
    tmp9 = tl.load(in_ptr3 + (r2 + (768*tmp8)), rmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 512
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.device_assert((0 <= tmp14) & (tmp14 < 512), "index out of bounds: 0 <= tmp14 < 512")
    tmp15 = tl.load(in_ptr5 + (r2 + (768*tmp14)), rmask, other=0.0)
    tmp16 = tmp10 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 768, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 768.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-12
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp16, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp43, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/am/camlpdmxoswydls5y4yae74wfiwbzgubk6x2wgh3dz6f4mxb2ylz.py
# Source Nodes: [add_2, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
# add_2 => add_5
# attention_output => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
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
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpyda6s2siljgrfc2vroltlfk4jndu56eo724fchhol65cjtrugf.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_8, erf, mul_5, mul_6, mul_7
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
    xnumel = 6291456
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


# kernel path: /tmp/torchinductor_youkaichao/nc/cncvnga2l7ji7bs4k46mjsx6oadgceblgb6bv6xau6kwphhsus6i.py
# Source Nodes: [hidden_states_109, hidden_states_111], Original ATen: [aten.gelu, aten.native_layer_norm]
# hidden_states_109 => add_100, erf_12, mul_87, mul_88, mul_89
# hidden_states_111 => add_101, add_102, mul_90, mul_91, rsqrt_25, sub_38, var_mean_25
triton_per_fused_gelu_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30522, 768), (768, 1))
    assert_size_stride(arg1_1, (2, 768), (768, 1))
    assert_size_stride(arg2_1, (512, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, 768), (768, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, 768), (768, 1))
    assert_size_stride(arg16_1, (3072, ), (1, ))
    assert_size_stride(arg17_1, (768, 3072), (3072, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, 768), (768, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, 768), (768, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768), (768, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (3072, ), (1, ))
    assert_size_stride(arg33_1, (768, 3072), (3072, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, 768), (768, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, 768), (768, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (3072, ), (1, ))
    assert_size_stride(arg49_1, (768, 3072), (3072, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, 768), (768, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 768), (768, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (3072, 768), (768, 1))
    assert_size_stride(arg64_1, (3072, ), (1, ))
    assert_size_stride(arg65_1, (768, 3072), (3072, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, 768), (768, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (3072, ), (1, ))
    assert_size_stride(arg81_1, (768, 3072), (3072, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, 768), (768, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 768), (768, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, 768), (768, 1))
    assert_size_stride(arg96_1, (3072, ), (1, ))
    assert_size_stride(arg97_1, (768, 3072), (3072, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, 768), (768, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, 768), (768, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (3072, 768), (768, 1))
    assert_size_stride(arg112_1, (3072, ), (1, ))
    assert_size_stride(arg113_1, (768, 3072), (3072, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, 768), (768, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, 768), (768, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, 768), (768, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (3072, 768), (768, 1))
    assert_size_stride(arg128_1, (3072, ), (1, ))
    assert_size_stride(arg129_1, (768, 3072), (3072, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, 768), (768, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, 768), (768, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (3072, 768), (768, 1))
    assert_size_stride(arg144_1, (3072, ), (1, ))
    assert_size_stride(arg145_1, (768, 3072), (3072, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, 768), (768, 1))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, 768), (768, 1))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, 768), (768, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (3072, 768), (768, 1))
    assert_size_stride(arg160_1, (3072, ), (1, ))
    assert_size_stride(arg161_1, (768, 3072), (3072, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, 768), (768, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, 768), (768, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, 768), (768, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, 768), (768, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (3072, 768), (768, 1))
    assert_size_stride(arg176_1, (3072, ), (1, ))
    assert_size_stride(arg177_1, (768, 3072), (3072, 1))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, 768), (768, 1))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, 768), (768, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 768), (768, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, 768), (768, 1))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (3072, 768), (768, 1))
    assert_size_stride(arg192_1, (3072, ), (1, ))
    assert_size_stride(arg193_1, (768, 3072), (3072, 1))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, 768), (768, 1))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (30522, 768), (768, 1))
    assert_size_stride(arg202_1, (30522, ), (1, ))
    assert_size_stride(arg203_1, (1, 512), (512, 1))
    assert_size_stride(arg204_1, (1, 512), (512, 1))
    assert_size_stride(arg205_1, (4, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg205_1, arg0_1, arg203_1, arg1_1, arg204_1, arg2_1, arg3_1, arg4_1, buf0, buf4, 2048, 768, grid=grid(2048), stream=stream0)
        del arg0_1
        del arg1_1
        del arg203_1
        del arg204_1
        del arg205_1
        del arg2_1
        del arg3_1
        del arg4_1
        buf5 = reinterpret_tensor(buf0, (2048, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg6_1, reinterpret_tensor(buf4, (2048, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
        del arg5_1
        del arg6_1
        buf6 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___bert_encoder_layer_0_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf4, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del arg7_1
        del arg8_1
        buf7 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___bert_encoder_layer_0_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del arg10_1
        del arg9_1
        # Source Nodes: [], Original ATen: []
        buf8 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf5, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf6, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf7, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf5
        buf9 = buf8[0]
        del buf8
        buf13 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), out=buf13)
        del arg11_1
        buf17 = reinterpret_tensor(buf9, (4, 512, 768), (393216, 768, 1), 0); del buf9  # reuse
        # Source Nodes: [add_2, attention_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf13, arg12_1, buf4, arg13_1, arg14_1, buf17, 2048, 768, grid=grid(2048), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        buf18 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf17, (2048, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 3072), (1, 768), 0), out=buf18)
        del arg15_1
        buf19 = reinterpret_tensor(buf18, (4, 512, 3072), (1572864, 3072, 1), 0); del buf18  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf19, arg16_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg16_1
        buf20 = reinterpret_tensor(buf4, (2048, 768), (768, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg17_1, (3072, 768), (1, 3072), 0), out=buf20)
        del arg17_1
        buf24 = reinterpret_tensor(buf13, (4, 512, 768), (393216, 768, 1), 0); del buf13  # reuse
        # Source Nodes: [add_3, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf20, arg18_1, buf17, arg19_1, arg20_1, buf24, 2048, 768, grid=grid(2048), stream=stream0)
        del arg18_1
        del arg19_1
        del arg20_1
        buf25 = buf20; del buf20  # reuse
        # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg22_1, reinterpret_tensor(buf24, (2048, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del arg21_1
        del arg22_1
        buf26 = reinterpret_tensor(buf17, (2048, 768), (768, 1), 0); del buf17  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_1_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg24_1, reinterpret_tensor(buf24, (2048, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf26)
        del arg23_1
        del arg24_1
        buf27 = buf6; del buf6  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_1_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg26_1, reinterpret_tensor(buf24, (2048, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
        del arg25_1
        del arg26_1
        # Source Nodes: [], Original ATen: []
        buf28 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf25, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf26, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf27, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf25
        buf29 = buf28[0]
        del buf28
        buf33 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (2048, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), out=buf33)
        del arg27_1
        buf37 = reinterpret_tensor(buf29, (4, 512, 768), (393216, 768, 1), 0); del buf29  # reuse
        # Source Nodes: [add_5, attention_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf33, arg28_1, buf24, arg29_1, arg30_1, buf37, 2048, 768, grid=grid(2048), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        buf38 = reinterpret_tensor(buf19, (2048, 3072), (3072, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (2048, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 3072), (1, 768), 0), out=buf38)
        del arg31_1
        buf39 = reinterpret_tensor(buf38, (4, 512, 3072), (1572864, 3072, 1), 0); del buf38  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf39, arg32_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg32_1
        buf40 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg33_1, (3072, 768), (1, 3072), 0), out=buf40)
        del arg33_1
        buf44 = buf24; del buf24  # reuse
        # Source Nodes: [add_6, hidden_states_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf40, arg34_1, buf37, arg35_1, arg36_1, buf44, 2048, 768, grid=grid(2048), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        buf45 = buf40; del buf40  # reuse
        # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg38_1, reinterpret_tensor(buf44, (2048, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf45)
        del arg37_1
        del arg38_1
        buf46 = reinterpret_tensor(buf37, (2048, 768), (768, 1), 0); del buf37  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_2_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg40_1, reinterpret_tensor(buf44, (2048, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf46)
        del arg39_1
        del arg40_1
        buf47 = buf26; del buf26  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_2_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg42_1, reinterpret_tensor(buf44, (2048, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
        del arg41_1
        del arg42_1
        # Source Nodes: [], Original ATen: []
        buf48 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf45, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf46, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf47, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf45
        buf49 = buf48[0]
        del buf48
        buf53 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (2048, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), out=buf53)
        del arg43_1
        buf57 = reinterpret_tensor(buf49, (4, 512, 768), (393216, 768, 1), 0); del buf49  # reuse
        # Source Nodes: [add_8, attention_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf53, arg44_1, buf44, arg45_1, arg46_1, buf57, 2048, 768, grid=grid(2048), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        buf58 = reinterpret_tensor(buf39, (2048, 3072), (3072, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (2048, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 3072), (1, 768), 0), out=buf58)
        del arg47_1
        buf59 = reinterpret_tensor(buf58, (4, 512, 3072), (1572864, 3072, 1), 0); del buf58  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf59, arg48_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg48_1
        buf60 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg49_1, (3072, 768), (1, 3072), 0), out=buf60)
        del arg49_1
        buf64 = buf44; del buf44  # reuse
        # Source Nodes: [add_9, hidden_states_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf60, arg50_1, buf57, arg51_1, arg52_1, buf64, 2048, 768, grid=grid(2048), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        buf65 = buf60; del buf60  # reuse
        # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg54_1, reinterpret_tensor(buf64, (2048, 768), (768, 1), 0), reinterpret_tensor(arg53_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf65)
        del arg53_1
        del arg54_1
        buf66 = reinterpret_tensor(buf57, (2048, 768), (768, 1), 0); del buf57  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_3_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg56_1, reinterpret_tensor(buf64, (2048, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf66)
        del arg55_1
        del arg56_1
        buf67 = buf46; del buf46  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_3_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg58_1, reinterpret_tensor(buf64, (2048, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf67)
        del arg57_1
        del arg58_1
        # Source Nodes: [], Original ATen: []
        buf68 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf65, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf66, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf67, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf65
        buf69 = buf68[0]
        del buf68
        buf73 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (2048, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), out=buf73)
        del arg59_1
        buf77 = reinterpret_tensor(buf69, (4, 512, 768), (393216, 768, 1), 0); del buf69  # reuse
        # Source Nodes: [add_11, attention_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf73, arg60_1, buf64, arg61_1, arg62_1, buf77, 2048, 768, grid=grid(2048), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        buf78 = reinterpret_tensor(buf59, (2048, 3072), (3072, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (2048, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 3072), (1, 768), 0), out=buf78)
        del arg63_1
        buf79 = reinterpret_tensor(buf78, (4, 512, 3072), (1572864, 3072, 1), 0); del buf78  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf79, arg64_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg64_1
        buf80 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg65_1, (3072, 768), (1, 3072), 0), out=buf80)
        del arg65_1
        buf84 = buf64; del buf64  # reuse
        # Source Nodes: [add_12, hidden_states_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf80, arg66_1, buf77, arg67_1, arg68_1, buf84, 2048, 768, grid=grid(2048), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        buf85 = buf80; del buf80  # reuse
        # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg70_1, reinterpret_tensor(buf84, (2048, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf85)
        del arg69_1
        del arg70_1
        buf86 = reinterpret_tensor(buf77, (2048, 768), (768, 1), 0); del buf77  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_4_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg72_1, reinterpret_tensor(buf84, (2048, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf86)
        del arg71_1
        del arg72_1
        buf87 = buf66; del buf66  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_4_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg74_1, reinterpret_tensor(buf84, (2048, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf87)
        del arg73_1
        del arg74_1
        # Source Nodes: [], Original ATen: []
        buf88 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf85, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf86, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf87, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf85
        buf89 = buf88[0]
        del buf88
        buf93 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf89, (2048, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), out=buf93)
        del arg75_1
        buf97 = reinterpret_tensor(buf89, (4, 512, 768), (393216, 768, 1), 0); del buf89  # reuse
        # Source Nodes: [add_14, attention_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf93, arg76_1, buf84, arg77_1, arg78_1, buf97, 2048, 768, grid=grid(2048), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        buf98 = reinterpret_tensor(buf79, (2048, 3072), (3072, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (2048, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 3072), (1, 768), 0), out=buf98)
        del arg79_1
        buf99 = reinterpret_tensor(buf98, (4, 512, 3072), (1572864, 3072, 1), 0); del buf98  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf99, arg80_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg80_1
        buf100 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg81_1, (3072, 768), (1, 3072), 0), out=buf100)
        del arg81_1
        buf104 = buf84; del buf84  # reuse
        # Source Nodes: [add_15, hidden_states_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf100, arg82_1, buf97, arg83_1, arg84_1, buf104, 2048, 768, grid=grid(2048), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        buf105 = reinterpret_tensor(buf97, (2048, 768), (768, 1), 0); del buf97  # reuse
        # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg86_1, reinterpret_tensor(buf104, (2048, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf105)
        del arg85_1
        del arg86_1
        buf106 = buf100; del buf100  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_5_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg88_1, reinterpret_tensor(buf104, (2048, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf106)
        del arg87_1
        del arg88_1
        buf107 = buf86; del buf86  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_5_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg90_1, reinterpret_tensor(buf104, (2048, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf107)
        del arg89_1
        del arg90_1
        # Source Nodes: [], Original ATen: []
        buf108 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf105, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf106, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf107, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf105
        buf109 = buf108[0]
        del buf108
        buf113 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (2048, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), out=buf113)
        del arg91_1
        buf117 = reinterpret_tensor(buf109, (4, 512, 768), (393216, 768, 1), 0); del buf109  # reuse
        # Source Nodes: [add_17, attention_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf113, arg92_1, buf104, arg93_1, arg94_1, buf117, 2048, 768, grid=grid(2048), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        buf118 = reinterpret_tensor(buf99, (2048, 3072), (3072, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (2048, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 3072), (1, 768), 0), out=buf118)
        del arg95_1
        buf119 = reinterpret_tensor(buf118, (4, 512, 3072), (1572864, 3072, 1), 0); del buf118  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf119, arg96_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg96_1
        buf120 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg97_1, (3072, 768), (1, 3072), 0), out=buf120)
        del arg97_1
        buf124 = buf104; del buf104  # reuse
        # Source Nodes: [add_18, hidden_states_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf120, arg98_1, buf117, arg99_1, arg100_1, buf124, 2048, 768, grid=grid(2048), stream=stream0)
        del arg100_1
        del arg98_1
        del arg99_1
        buf125 = buf120; del buf120  # reuse
        # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg102_1, reinterpret_tensor(buf124, (2048, 768), (768, 1), 0), reinterpret_tensor(arg101_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf125)
        del arg101_1
        del arg102_1
        buf126 = reinterpret_tensor(buf117, (2048, 768), (768, 1), 0); del buf117  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_6_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg104_1, reinterpret_tensor(buf124, (2048, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf126)
        del arg103_1
        del arg104_1
        buf127 = buf106; del buf106  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_6_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg106_1, reinterpret_tensor(buf124, (2048, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf127)
        del arg105_1
        del arg106_1
        # Source Nodes: [], Original ATen: []
        buf128 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf125, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf126, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf127, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf125
        buf129 = buf128[0]
        del buf128
        buf133 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (2048, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), out=buf133)
        del arg107_1
        buf137 = reinterpret_tensor(buf129, (4, 512, 768), (393216, 768, 1), 0); del buf129  # reuse
        # Source Nodes: [add_20, attention_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf133, arg108_1, buf124, arg109_1, arg110_1, buf137, 2048, 768, grid=grid(2048), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        buf138 = reinterpret_tensor(buf119, (2048, 3072), (3072, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (2048, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 3072), (1, 768), 0), out=buf138)
        del arg111_1
        buf139 = reinterpret_tensor(buf138, (4, 512, 3072), (1572864, 3072, 1), 0); del buf138  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf139, arg112_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg112_1
        buf140 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg113_1, (3072, 768), (1, 3072), 0), out=buf140)
        del arg113_1
        buf144 = buf124; del buf124  # reuse
        # Source Nodes: [add_21, hidden_states_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf140, arg114_1, buf137, arg115_1, arg116_1, buf144, 2048, 768, grid=grid(2048), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        buf145 = buf140; del buf140  # reuse
        # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg118_1, reinterpret_tensor(buf144, (2048, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf145)
        del arg117_1
        del arg118_1
        buf146 = reinterpret_tensor(buf137, (2048, 768), (768, 1), 0); del buf137  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_7_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg120_1, reinterpret_tensor(buf144, (2048, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf146)
        del arg119_1
        del arg120_1
        buf147 = buf126; del buf126  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_7_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg122_1, reinterpret_tensor(buf144, (2048, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf147)
        del arg121_1
        del arg122_1
        # Source Nodes: [], Original ATen: []
        buf148 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf145, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf146, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf147, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf145
        buf149 = buf148[0]
        del buf148
        buf153 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (2048, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), out=buf153)
        del arg123_1
        buf157 = reinterpret_tensor(buf149, (4, 512, 768), (393216, 768, 1), 0); del buf149  # reuse
        # Source Nodes: [add_23, attention_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf153, arg124_1, buf144, arg125_1, arg126_1, buf157, 2048, 768, grid=grid(2048), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        buf158 = reinterpret_tensor(buf139, (2048, 3072), (3072, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (2048, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 3072), (1, 768), 0), out=buf158)
        del arg127_1
        buf159 = reinterpret_tensor(buf158, (4, 512, 3072), (1572864, 3072, 1), 0); del buf158  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf159, arg128_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg128_1
        buf160 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg129_1, (3072, 768), (1, 3072), 0), out=buf160)
        del arg129_1
        buf164 = buf144; del buf144  # reuse
        # Source Nodes: [add_24, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf160, arg130_1, buf157, arg131_1, arg132_1, buf164, 2048, 768, grid=grid(2048), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        buf165 = buf160; del buf160  # reuse
        # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg134_1, reinterpret_tensor(buf164, (2048, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf165)
        del arg133_1
        del arg134_1
        buf166 = reinterpret_tensor(buf157, (2048, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_8_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg136_1, reinterpret_tensor(buf164, (2048, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf166)
        del arg135_1
        del arg136_1
        buf167 = buf146; del buf146  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_8_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg138_1, reinterpret_tensor(buf164, (2048, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf167)
        del arg137_1
        del arg138_1
        # Source Nodes: [], Original ATen: []
        buf168 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf165, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf166, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf167, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf165
        buf169 = buf168[0]
        del buf168
        buf173 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (2048, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), out=buf173)
        del arg139_1
        buf177 = reinterpret_tensor(buf169, (4, 512, 768), (393216, 768, 1), 0); del buf169  # reuse
        # Source Nodes: [add_26, attention_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf173, arg140_1, buf164, arg141_1, arg142_1, buf177, 2048, 768, grid=grid(2048), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        buf178 = reinterpret_tensor(buf159, (2048, 3072), (3072, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (2048, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 3072), (1, 768), 0), out=buf178)
        del arg143_1
        buf179 = reinterpret_tensor(buf178, (4, 512, 3072), (1572864, 3072, 1), 0); del buf178  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf179, arg144_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg144_1
        buf180 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg145_1, (3072, 768), (1, 3072), 0), out=buf180)
        del arg145_1
        buf184 = buf164; del buf164  # reuse
        # Source Nodes: [add_27, hidden_states_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf180, arg146_1, buf177, arg147_1, arg148_1, buf184, 2048, 768, grid=grid(2048), stream=stream0)
        del arg146_1
        del arg147_1
        del arg148_1
        buf185 = buf180; del buf180  # reuse
        # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg150_1, reinterpret_tensor(buf184, (2048, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf185)
        del arg149_1
        del arg150_1
        buf186 = reinterpret_tensor(buf177, (2048, 768), (768, 1), 0); del buf177  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_9_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg152_1, reinterpret_tensor(buf184, (2048, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf186)
        del arg151_1
        del arg152_1
        buf187 = buf166; del buf166  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_9_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg154_1, reinterpret_tensor(buf184, (2048, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf187)
        del arg153_1
        del arg154_1
        # Source Nodes: [], Original ATen: []
        buf188 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf185, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf186, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf187, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf185
        buf189 = buf188[0]
        del buf188
        buf193 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (2048, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), out=buf193)
        del arg155_1
        buf197 = reinterpret_tensor(buf189, (4, 512, 768), (393216, 768, 1), 0); del buf189  # reuse
        # Source Nodes: [add_29, attention_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf193, arg156_1, buf184, arg157_1, arg158_1, buf197, 2048, 768, grid=grid(2048), stream=stream0)
        del arg156_1
        del arg157_1
        del arg158_1
        buf198 = reinterpret_tensor(buf179, (2048, 3072), (3072, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (2048, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 3072), (1, 768), 0), out=buf198)
        del arg159_1
        buf199 = reinterpret_tensor(buf198, (4, 512, 3072), (1572864, 3072, 1), 0); del buf198  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf199, arg160_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg160_1
        buf200 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg161_1, (3072, 768), (1, 3072), 0), out=buf200)
        del arg161_1
        buf204 = buf184; del buf184  # reuse
        # Source Nodes: [add_30, hidden_states_89], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf200, arg162_1, buf197, arg163_1, arg164_1, buf204, 2048, 768, grid=grid(2048), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        buf205 = buf200; del buf200  # reuse
        # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg166_1, reinterpret_tensor(buf204, (2048, 768), (768, 1), 0), reinterpret_tensor(arg165_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf205)
        del arg165_1
        del arg166_1
        buf206 = reinterpret_tensor(buf197, (2048, 768), (768, 1), 0); del buf197  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_10_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg168_1, reinterpret_tensor(buf204, (2048, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf206)
        del arg167_1
        del arg168_1
        buf207 = buf186; del buf186  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_10_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg170_1, reinterpret_tensor(buf204, (2048, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf207)
        del arg169_1
        del arg170_1
        # Source Nodes: [], Original ATen: []
        buf208 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf205, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf206, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf207, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf205
        buf209 = buf208[0]
        del buf208
        buf213 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (2048, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), out=buf213)
        del arg171_1
        buf217 = reinterpret_tensor(buf209, (4, 512, 768), (393216, 768, 1), 0); del buf209  # reuse
        # Source Nodes: [add_32, attention_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf213, arg172_1, buf204, arg173_1, arg174_1, buf217, 2048, 768, grid=grid(2048), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        buf218 = reinterpret_tensor(buf199, (2048, 3072), (3072, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (2048, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 3072), (1, 768), 0), out=buf218)
        del arg175_1
        buf219 = reinterpret_tensor(buf218, (4, 512, 3072), (1572864, 3072, 1), 0); del buf218  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf219, arg176_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg176_1
        buf220 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg177_1, (3072, 768), (1, 3072), 0), out=buf220)
        del arg177_1
        buf224 = buf204; del buf204  # reuse
        # Source Nodes: [add_33, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf220, arg178_1, buf217, arg179_1, arg180_1, buf224, 2048, 768, grid=grid(2048), stream=stream0)
        del arg178_1
        del arg179_1
        del arg180_1
        buf225 = buf220; del buf220  # reuse
        # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg182_1, reinterpret_tensor(buf224, (2048, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf225)
        del arg181_1
        del arg182_1
        buf226 = reinterpret_tensor(buf217, (2048, 768), (768, 1), 0); del buf217  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_11_attention_self_key], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg184_1, reinterpret_tensor(buf224, (2048, 768), (768, 1), 0), reinterpret_tensor(arg183_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf226)
        del arg183_1
        del arg184_1
        buf227 = buf206; del buf206  # reuse
        # Source Nodes: [l__mod___bert_encoder_layer_11_attention_self_value], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg186_1, reinterpret_tensor(buf224, (2048, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf227)
        del arg185_1
        del arg186_1
        # Source Nodes: [], Original ATen: []
        buf228 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf225, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf226, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf227, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf225
        del buf226
        buf229 = buf228[0]
        del buf228
        buf233 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (2048, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), out=buf233)
        del arg187_1
        buf237 = reinterpret_tensor(buf229, (4, 512, 768), (393216, 768, 1), 0); del buf229  # reuse
        # Source Nodes: [add_35, attention_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf233, arg188_1, buf224, arg189_1, arg190_1, buf237, 2048, 768, grid=grid(2048), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        buf238 = reinterpret_tensor(buf219, (2048, 3072), (3072, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (2048, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 3072), (1, 768), 0), out=buf238)
        del arg191_1
        buf239 = reinterpret_tensor(buf238, (4, 512, 3072), (1572864, 3072, 1), 0); del buf238  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf239, arg192_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg192_1
        buf240 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg193_1, (3072, 768), (1, 3072), 0), out=buf240)
        del arg193_1
        del buf239
        buf244 = buf224; del buf224  # reuse
        # Source Nodes: [add_36, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf240, arg194_1, buf237, arg195_1, arg196_1, buf244, 2048, 768, grid=grid(2048), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        del buf237
        buf245 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (2048, 768), (768, 1), 0), reinterpret_tensor(arg197_1, (768, 768), (1, 768), 0), out=buf245)
        del arg197_1
        buf249 = buf244; del buf244  # reuse
        # Source Nodes: [hidden_states_109, hidden_states_111], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_3.run(buf245, arg198_1, arg199_1, arg200_1, buf249, 2048, 768, grid=grid(2048), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del buf245
        buf250 = empty((2048, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg202_1, reinterpret_tensor(buf249, (2048, 768), (768, 1), 0), reinterpret_tensor(arg201_1, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf250)
        del arg201_1
        del arg202_1
        return (reinterpret_tensor(buf250, (4, 512, 30522), (15627264, 30522, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg204_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg205_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Bert', benchmark_compiled_module)
