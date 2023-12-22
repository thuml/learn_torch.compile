
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


# kernel path: /tmp/torchinductor_youkaichao/5o/c5ojjuy3kn77jea3jco6bt3wdcyrrqojiuyeauzmyrsqn525zphz.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 30000
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 30000), "index out of bounds: 0 <= tmp3 < 30000")
    tmp4 = tl.load(in_ptr1 + (r2 + (128*tmp3)), rmask, other=0.0)
    tmp6 = tmp5 + 2
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert((0 <= tmp8) & (tmp8 < 2), "index out of bounds: 0 <= tmp8 < 2")
    tmp9 = tl.load(in_ptr3 + (r2 + (128*tmp8)), rmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 512
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.device_assert((0 <= tmp14) & (tmp14 < 512), "index out of bounds: 0 <= tmp14 < 512")
    tmp15 = tl.load(in_ptr5 + (r2 + (128*tmp14)), rmask, other=0.0)
    tmp16 = tmp10 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp16 - tmp26
    tmp34 = 128.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-12
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp16, rmask)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp43, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/b2/cb2klaaabalcrkzb6sxbvkrtzvkz3n3dgsdexnmahsmgvq6rvnku.py
# Source Nodes: [add_2, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm]
# add_2 => add_5
# layernormed_context_layer => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6nfimj4vrxsc5sxrxmxmij7n6mrdbieiqus3xqsnfsqwh7f435.py
# Source Nodes: [add_3, add_4, ffn_output_1, mul_1, mul_2, mul_3, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_3 => add_8
# add_4 => add_9
# ffn_output_1 => mul_8
# mul_1 => mul_5
# mul_2 => mul_6
# mul_3 => mul_7
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_2', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = tl.math.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrlt3spy5koufmemeszuwikbs2msmuw3bopb24bflibn3ie3hyo.py
# Source Nodes: [add_5, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
# add_5 => add_10
# hidden_states_3 => add_11, add_12, mul_10, mul_9, rsqrt_2, sub_4, var_mean_2
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6n/c6ntxrbmonkk7f6zl3v77shgnev32epnsdlpjmtmvnfb2nispsen.py
# Source Nodes: [add_61, add_62, hidden_states_38, hidden_states_39, mul_49, mul_50, mul_51, pow_13, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh]
# add_61 => add_112
# add_62 => add_113
# hidden_states_38 => mul_102
# hidden_states_39 => add_114, add_115, mul_103, mul_104, rsqrt_25, sub_38, var_mean_25
# mul_49 => mul_99
# mul_50 => mul_100
# mul_51 => mul_101
# pow_13 => pow_13
# tanh_12 => tanh_12
triton_per_fused_add_mul_native_layer_norm_pow_tanh_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_pow_tanh_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = tl.math.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = tmp15 - tmp25
    tmp33 = 128.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-12
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp42, rmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1 = args
    args.clear()
    assert_size_stride(arg0_1, (30000, 128), (128, 1))
    assert_size_stride(arg1_1, (2, 128), (128, 1))
    assert_size_stride(arg2_1, (512, 128), (128, 1))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (768, 128), (128, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, 768), (768, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, 768), (768, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (3072, 768), (768, 1))
    assert_size_stride(arg18_1, (3072, ), (1, ))
    assert_size_stride(arg19_1, (768, 3072), (3072, 1))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (128, 768), (768, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (30000, 128), (128, 1))
    assert_size_stride(arg28_1, (30000, ), (1, ))
    assert_size_stride(arg29_1, (1, 512), (512, 1))
    assert_size_stride(arg30_1, (1, 512), (512, 1))
    assert_size_stride(arg31_1, (4, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 512, 128), device='cuda', dtype=torch.float32)
        buf4 = empty((4, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg31_1, arg0_1, arg29_1, arg1_1, arg30_1, arg2_1, arg3_1, arg4_1, buf0, buf4, 2048, 128, grid=grid(2048), stream=stream0)
        del arg0_1
        del arg1_1
        del arg29_1
        del arg2_1
        del arg30_1
        del arg31_1
        del arg3_1
        del arg4_1
        buf5 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg6_1, reinterpret_tensor(buf4, (2048, 128), (128, 1), 0), reinterpret_tensor(arg5_1, (128, 768), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg5_1
        del arg6_1
        buf6 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf5, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
        buf7 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_key_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf5, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        buf8 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [mixed_value_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf5, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf8)
        # Source Nodes: [], Original ATen: []
        buf9 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf6, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf7, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf8, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf6
        buf10 = buf9[0]
        del buf9
        buf14 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf10, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf14)
        buf18 = reinterpret_tensor(buf10, (4, 512, 768), (393216, 768, 1), 0); del buf10  # reuse
        # Source Nodes: [add_2, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf5, buf14, arg14_1, arg15_1, arg16_1, buf18, 2048, 768, grid=grid(2048), stream=stream0)
        buf19 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf19)
        buf20 = reinterpret_tensor(buf19, (4, 512, 3072), (1572864, 3072, 1), 0); del buf19  # reuse
        # Source Nodes: [add_3, add_4, ffn_output_1, mul_1, mul_2, mul_3, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf20, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf21 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf21)
        buf25 = reinterpret_tensor(buf14, (4, 512, 768), (393216, 768, 1), 0); del buf14  # reuse
        # Source Nodes: [add_5, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf21, arg20_1, buf18, arg21_1, arg22_1, buf25, 2048, 768, grid=grid(2048), stream=stream0)
        buf26 = buf21; del buf21  # reuse
        # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf25, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf26)
        buf27 = reinterpret_tensor(buf18, (2048, 768), (768, 1), 0); del buf18  # reuse
        # Source Nodes: [mixed_key_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf25, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
        buf28 = buf7; del buf7  # reuse
        # Source Nodes: [mixed_value_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf25, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
        # Source Nodes: [], Original ATen: []
        buf29 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf26, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf27, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf28, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf26
        buf30 = buf29[0]
        del buf29
        buf34 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf34)
        buf38 = reinterpret_tensor(buf30, (4, 512, 768), (393216, 768, 1), 0); del buf30  # reuse
        # Source Nodes: [add_7, layernormed_context_layer_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf25, buf34, arg14_1, arg15_1, arg16_1, buf38, 2048, 768, grid=grid(2048), stream=stream0)
        buf39 = reinterpret_tensor(buf20, (2048, 3072), (3072, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf39)
        buf40 = reinterpret_tensor(buf39, (4, 512, 3072), (1572864, 3072, 1), 0); del buf39  # reuse
        # Source Nodes: [add_8, add_9, ffn_output_5, mul_5, mul_6, mul_7, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf40, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf41 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf41)
        buf45 = buf25; del buf25  # reuse
        # Source Nodes: [add_10, hidden_states_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf41, arg20_1, buf38, arg21_1, arg22_1, buf45, 2048, 768, grid=grid(2048), stream=stream0)
        buf46 = buf41; del buf41  # reuse
        # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf45, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf46)
        buf47 = reinterpret_tensor(buf38, (2048, 768), (768, 1), 0); del buf38  # reuse
        # Source Nodes: [mixed_key_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf45, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
        buf48 = buf27; del buf27  # reuse
        # Source Nodes: [mixed_value_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf45, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf48)
        # Source Nodes: [], Original ATen: []
        buf49 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf46, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf47, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf48, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf46
        buf50 = buf49[0]
        del buf49
        buf54 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf54)
        buf58 = reinterpret_tensor(buf50, (4, 512, 768), (393216, 768, 1), 0); del buf50  # reuse
        # Source Nodes: [add_12, layernormed_context_layer_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf45, buf54, arg14_1, arg15_1, arg16_1, buf58, 2048, 768, grid=grid(2048), stream=stream0)
        buf59 = reinterpret_tensor(buf40, (2048, 3072), (3072, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf59)
        buf60 = reinterpret_tensor(buf59, (4, 512, 3072), (1572864, 3072, 1), 0); del buf59  # reuse
        # Source Nodes: [add_13, add_14, ffn_output_9, mul_10, mul_11, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf60, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf61 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf61)
        buf65 = buf45; del buf45  # reuse
        # Source Nodes: [add_15, hidden_states_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf61, arg20_1, buf58, arg21_1, arg22_1, buf65, 2048, 768, grid=grid(2048), stream=stream0)
        buf66 = buf61; del buf61  # reuse
        # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf65, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf66)
        buf67 = reinterpret_tensor(buf58, (2048, 768), (768, 1), 0); del buf58  # reuse
        # Source Nodes: [mixed_key_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf65, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf67)
        buf68 = buf47; del buf47  # reuse
        # Source Nodes: [mixed_value_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf65, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf68)
        # Source Nodes: [], Original ATen: []
        buf69 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf66, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf67, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf68, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf66
        buf70 = buf69[0]
        del buf69
        buf74 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf74)
        buf78 = reinterpret_tensor(buf70, (4, 512, 768), (393216, 768, 1), 0); del buf70  # reuse
        # Source Nodes: [add_17, layernormed_context_layer_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf65, buf74, arg14_1, arg15_1, arg16_1, buf78, 2048, 768, grid=grid(2048), stream=stream0)
        buf79 = reinterpret_tensor(buf60, (2048, 3072), (3072, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf79)
        buf80 = reinterpret_tensor(buf79, (4, 512, 3072), (1572864, 3072, 1), 0); del buf79  # reuse
        # Source Nodes: [add_18, add_19, ffn_output_13, mul_13, mul_14, mul_15, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf80, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf81 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf81)
        buf85 = buf65; del buf65  # reuse
        # Source Nodes: [add_20, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf81, arg20_1, buf78, arg21_1, arg22_1, buf85, 2048, 768, grid=grid(2048), stream=stream0)
        buf86 = buf81; del buf81  # reuse
        # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf85, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf86)
        buf87 = reinterpret_tensor(buf78, (2048, 768), (768, 1), 0); del buf78  # reuse
        # Source Nodes: [mixed_key_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf85, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf87)
        buf88 = buf67; del buf67  # reuse
        # Source Nodes: [mixed_value_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf85, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf88)
        # Source Nodes: [], Original ATen: []
        buf89 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf86, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf87, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf88, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf86
        buf90 = buf89[0]
        del buf89
        buf94 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf94)
        buf98 = reinterpret_tensor(buf90, (4, 512, 768), (393216, 768, 1), 0); del buf90  # reuse
        # Source Nodes: [add_22, layernormed_context_layer_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf85, buf94, arg14_1, arg15_1, arg16_1, buf98, 2048, 768, grid=grid(2048), stream=stream0)
        buf99 = reinterpret_tensor(buf80, (2048, 3072), (3072, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf99)
        buf100 = reinterpret_tensor(buf99, (4, 512, 3072), (1572864, 3072, 1), 0); del buf99  # reuse
        # Source Nodes: [add_23, add_24, ffn_output_17, mul_17, mul_18, mul_19, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf100, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf101 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf101)
        buf105 = buf85; del buf85  # reuse
        # Source Nodes: [add_25, hidden_states_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf101, arg20_1, buf98, arg21_1, arg22_1, buf105, 2048, 768, grid=grid(2048), stream=stream0)
        buf106 = reinterpret_tensor(buf98, (2048, 768), (768, 1), 0); del buf98  # reuse
        # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf105, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf106)
        buf107 = buf101; del buf101  # reuse
        # Source Nodes: [mixed_key_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf105, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf107)
        buf108 = buf87; del buf87  # reuse
        # Source Nodes: [mixed_value_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf105, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf108)
        # Source Nodes: [], Original ATen: []
        buf109 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf106, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf107, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf108, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf106
        buf110 = buf109[0]
        del buf109
        buf114 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf114)
        buf118 = reinterpret_tensor(buf110, (4, 512, 768), (393216, 768, 1), 0); del buf110  # reuse
        # Source Nodes: [add_27, layernormed_context_layer_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf105, buf114, arg14_1, arg15_1, arg16_1, buf118, 2048, 768, grid=grid(2048), stream=stream0)
        buf119 = reinterpret_tensor(buf100, (2048, 3072), (3072, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf119)
        buf120 = reinterpret_tensor(buf119, (4, 512, 3072), (1572864, 3072, 1), 0); del buf119  # reuse
        # Source Nodes: [add_28, add_29, ffn_output_21, mul_21, mul_22, mul_23, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf120, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf121 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf121)
        buf125 = buf105; del buf105  # reuse
        # Source Nodes: [add_30, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf121, arg20_1, buf118, arg21_1, arg22_1, buf125, 2048, 768, grid=grid(2048), stream=stream0)
        buf126 = buf121; del buf121  # reuse
        # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf125, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf126)
        buf127 = reinterpret_tensor(buf118, (2048, 768), (768, 1), 0); del buf118  # reuse
        # Source Nodes: [mixed_key_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf125, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf127)
        buf128 = buf107; del buf107  # reuse
        # Source Nodes: [mixed_value_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf125, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf128)
        # Source Nodes: [], Original ATen: []
        buf129 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf126, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf127, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf128, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf126
        buf130 = buf129[0]
        del buf129
        buf134 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf134)
        buf138 = reinterpret_tensor(buf130, (4, 512, 768), (393216, 768, 1), 0); del buf130  # reuse
        # Source Nodes: [add_32, layernormed_context_layer_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf125, buf134, arg14_1, arg15_1, arg16_1, buf138, 2048, 768, grid=grid(2048), stream=stream0)
        buf139 = reinterpret_tensor(buf120, (2048, 3072), (3072, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf139)
        buf140 = reinterpret_tensor(buf139, (4, 512, 3072), (1572864, 3072, 1), 0); del buf139  # reuse
        # Source Nodes: [add_33, add_34, ffn_output_25, mul_25, mul_26, mul_27, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf140, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf141 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf141)
        buf145 = buf125; del buf125  # reuse
        # Source Nodes: [add_35, hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf141, arg20_1, buf138, arg21_1, arg22_1, buf145, 2048, 768, grid=grid(2048), stream=stream0)
        buf146 = buf141; del buf141  # reuse
        # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf145, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf146)
        buf147 = reinterpret_tensor(buf138, (2048, 768), (768, 1), 0); del buf138  # reuse
        # Source Nodes: [mixed_key_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf145, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf147)
        buf148 = buf127; del buf127  # reuse
        # Source Nodes: [mixed_value_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf145, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf148)
        # Source Nodes: [], Original ATen: []
        buf149 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf146, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf147, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf148, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf146
        buf150 = buf149[0]
        del buf149
        buf154 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf154)
        buf158 = reinterpret_tensor(buf150, (4, 512, 768), (393216, 768, 1), 0); del buf150  # reuse
        # Source Nodes: [add_37, layernormed_context_layer_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf145, buf154, arg14_1, arg15_1, arg16_1, buf158, 2048, 768, grid=grid(2048), stream=stream0)
        buf159 = reinterpret_tensor(buf140, (2048, 3072), (3072, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf159)
        buf160 = reinterpret_tensor(buf159, (4, 512, 3072), (1572864, 3072, 1), 0); del buf159  # reuse
        # Source Nodes: [add_38, add_39, ffn_output_29, mul_29, mul_30, mul_31, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf160, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf161 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf161)
        buf165 = buf145; del buf145  # reuse
        # Source Nodes: [add_40, hidden_states_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf161, arg20_1, buf158, arg21_1, arg22_1, buf165, 2048, 768, grid=grid(2048), stream=stream0)
        buf166 = buf161; del buf161  # reuse
        # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf165, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf166)
        buf167 = reinterpret_tensor(buf158, (2048, 768), (768, 1), 0); del buf158  # reuse
        # Source Nodes: [mixed_key_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf165, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf167)
        buf168 = buf147; del buf147  # reuse
        # Source Nodes: [mixed_value_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf165, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf168)
        # Source Nodes: [], Original ATen: []
        buf169 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf166, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf167, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf168, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf166
        buf170 = buf169[0]
        del buf169
        buf174 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf174)
        buf178 = reinterpret_tensor(buf170, (4, 512, 768), (393216, 768, 1), 0); del buf170  # reuse
        # Source Nodes: [add_42, layernormed_context_layer_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf165, buf174, arg14_1, arg15_1, arg16_1, buf178, 2048, 768, grid=grid(2048), stream=stream0)
        buf179 = reinterpret_tensor(buf160, (2048, 3072), (3072, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf179)
        buf180 = reinterpret_tensor(buf179, (4, 512, 3072), (1572864, 3072, 1), 0); del buf179  # reuse
        # Source Nodes: [add_43, add_44, ffn_output_33, mul_33, mul_34, mul_35, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf180, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf181 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf181)
        buf185 = buf165; del buf165  # reuse
        # Source Nodes: [add_45, hidden_states_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf181, arg20_1, buf178, arg21_1, arg22_1, buf185, 2048, 768, grid=grid(2048), stream=stream0)
        buf186 = buf181; del buf181  # reuse
        # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf185, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf186)
        buf187 = reinterpret_tensor(buf178, (2048, 768), (768, 1), 0); del buf178  # reuse
        # Source Nodes: [mixed_key_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf185, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf187)
        buf188 = buf167; del buf167  # reuse
        # Source Nodes: [mixed_value_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf185, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf188)
        # Source Nodes: [], Original ATen: []
        buf189 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf186, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf187, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf188, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf186
        buf190 = buf189[0]
        del buf189
        buf194 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf194)
        buf198 = reinterpret_tensor(buf190, (4, 512, 768), (393216, 768, 1), 0); del buf190  # reuse
        # Source Nodes: [add_47, layernormed_context_layer_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf185, buf194, arg14_1, arg15_1, arg16_1, buf198, 2048, 768, grid=grid(2048), stream=stream0)
        buf199 = reinterpret_tensor(buf180, (2048, 3072), (3072, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf199)
        buf200 = reinterpret_tensor(buf199, (4, 512, 3072), (1572864, 3072, 1), 0); del buf199  # reuse
        # Source Nodes: [add_48, add_49, ffn_output_37, mul_37, mul_38, mul_39, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf200, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf201 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf201)
        buf205 = buf185; del buf185  # reuse
        # Source Nodes: [add_50, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf201, arg20_1, buf198, arg21_1, arg22_1, buf205, 2048, 768, grid=grid(2048), stream=stream0)
        buf206 = buf201; del buf201  # reuse
        # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf205, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf206)
        buf207 = reinterpret_tensor(buf198, (2048, 768), (768, 1), 0); del buf198  # reuse
        # Source Nodes: [mixed_key_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf205, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf207)
        buf208 = buf187; del buf187  # reuse
        # Source Nodes: [mixed_value_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf205, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf208)
        # Source Nodes: [], Original ATen: []
        buf209 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf206, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf207, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf208, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf206
        buf210 = buf209[0]
        del buf209
        buf214 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf214)
        buf218 = reinterpret_tensor(buf210, (4, 512, 768), (393216, 768, 1), 0); del buf210  # reuse
        # Source Nodes: [add_52, layernormed_context_layer_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf205, buf214, arg14_1, arg15_1, arg16_1, buf218, 2048, 768, grid=grid(2048), stream=stream0)
        buf219 = reinterpret_tensor(buf200, (2048, 3072), (3072, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf219)
        buf220 = reinterpret_tensor(buf219, (4, 512, 3072), (1572864, 3072, 1), 0); del buf219  # reuse
        # Source Nodes: [add_53, add_54, ffn_output_41, mul_41, mul_42, mul_43, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf220, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        buf221 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf221)
        buf225 = buf205; del buf205  # reuse
        # Source Nodes: [add_55, hidden_states_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf221, arg20_1, buf218, arg21_1, arg22_1, buf225, 2048, 768, grid=grid(2048), stream=stream0)
        buf226 = buf221; del buf221  # reuse
        # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf225, (2048, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf226)
        del arg7_1
        del arg8_1
        buf227 = reinterpret_tensor(buf218, (2048, 768), (768, 1), 0); del buf218  # reuse
        # Source Nodes: [mixed_key_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf225, (2048, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf227)
        del arg10_1
        del arg9_1
        buf228 = buf207; del buf207  # reuse
        # Source Nodes: [mixed_value_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf225, (2048, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf228)
        del arg11_1
        del arg12_1
        # Source Nodes: [], Original ATen: []
        buf229 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf226, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf227, (4, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf228, (4, 12, 512, 64), (393216, 64, 768, 1), 0), None, True, scale=0.125)
        del buf226
        del buf227
        buf230 = buf229[0]
        del buf229
        buf234 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (2048, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf234)
        del arg13_1
        buf238 = reinterpret_tensor(buf230, (4, 512, 768), (393216, 768, 1), 0); del buf230  # reuse
        # Source Nodes: [add_57, layernormed_context_layer_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf225, buf234, arg14_1, arg15_1, arg16_1, buf238, 2048, 768, grid=grid(2048), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        buf239 = reinterpret_tensor(buf220, (2048, 3072), (3072, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (2048, 768), (768, 1), 0), reinterpret_tensor(arg17_1, (768, 3072), (1, 768), 0), out=buf239)
        del arg17_1
        buf240 = reinterpret_tensor(buf239, (4, 512, 3072), (1572864, 3072, 1), 0); del buf239  # reuse
        # Source Nodes: [add_58, add_59, ffn_output_45, mul_45, mul_46, mul_47, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_2.run(buf240, arg18_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg18_1
        buf241 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (2048, 3072), (3072, 1), 0), reinterpret_tensor(arg19_1, (3072, 768), (1, 3072), 0), out=buf241)
        del arg19_1
        del buf240
        buf245 = buf225; del buf225  # reuse
        # Source Nodes: [add_60, sequence_outputs], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf241, arg20_1, buf238, arg21_1, arg22_1, buf245, 2048, 768, grid=grid(2048), stream=stream0)
        del arg20_1
        del arg21_1
        del arg22_1
        del buf238
        del buf241
        buf246 = reinterpret_tensor(buf4, (2048, 128), (128, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (2048, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 128), (1, 768), 0), out=buf246)
        del arg23_1
        del buf245
        buf250 = buf0; del buf0  # reuse
        # Source Nodes: [add_61, add_62, hidden_states_38, hidden_states_39, mul_49, mul_50, mul_51, pow_13, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh]
        triton_per_fused_add_mul_native_layer_norm_pow_tanh_4.run(buf246, arg24_1, arg25_1, arg26_1, buf250, 2048, 128, grid=grid(2048), stream=stream0)
        del arg24_1
        del arg25_1
        del arg26_1
        del buf246
        buf251 = empty((2048, 30000), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg28_1, reinterpret_tensor(buf250, (2048, 128), (128, 1), 0), reinterpret_tensor(arg27_1, (128, 30000), (1, 128), 0), alpha=1, beta=1, out=buf251)
        del arg27_1
        del arg28_1
        return (reinterpret_tensor(buf251, (4, 512, 30000), (15360000, 30000, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((30000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg30_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg31_1 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Albert', benchmark_compiled_module)
