
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


# kernel path: /tmp/torchinductor_youkaichao/uo/cuo7ho2rrrcmwbf4dwxuwwl2nebsjxlp6xcbd6wngf5vjvenkvxp.py
# Source Nodes: [embeddings, embeddings_1, inputs_embeds, ln_outputs, position_embeddings, token_type_embeddings, token_type_ids], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.zeros]
# embeddings => add
# embeddings_1 => add_1
# inputs_embeds => embedding
# ln_outputs => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# position_embeddings => embedding_2
# token_type_embeddings => embedding_1
# token_type_ids => full_default
triton_per_fused_add_embedding_native_layer_norm_zeros_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_zeros_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 29056
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 29056)) | ~xmask, "index out of bounds: 0 <= tmp3 < 29056")
    tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7 + 512
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 512)) | ~xmask, "index out of bounds: 0 <= tmp10 < 512")
    tmp11 = tl.load(in_ptr4 + (r1 + (1024*tmp10)), rmask & xmask, other=0.0)
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 1024, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 1024.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp12, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp39, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhvu7fsvf2c5gku2a2tl7bsjhtdsyhtooh7bbry5fvso4cqqdc2.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cacko5ionofjuojyp5cpqidw7ers7r4qalizuhv675hnxfczzat3.py
# Source Nodes: [attention_output, ln_output], Original ATen: [aten.add, aten.native_layer_norm]
# attention_output => add_5
# ln_output => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
triton_per_fused_add_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 1024.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwed2ttuuhf6ibkx3bmodg4rhc2wtr2ix4irdjkt3p4edhd4dxj.py
# Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
# intermediate_output => add_8, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
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


# kernel path: /tmp/torchinductor_youkaichao/rd/crdp6j7bxuyo4ubbp77gxs7jb435sbvzijxkkpzaxr5tcnr2ndxh.py
# Source Nodes: [attention_output, hidden_states_6, ln_outputs_1], Original ATen: [aten.add, aten.native_layer_norm]
# attention_output => add_5
# hidden_states_6 => add_9
# ln_outputs_1 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_4, var_mean_2
triton_per_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 1024.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-12
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbes7axkjrun6wshw5eded647hz6q5kzugrkzxxoqgtpbqclm6eb.py
# Source Nodes: [hidden_states_170, hidden_states_172], Original ATen: [aten.gelu, aten.native_layer_norm]
# hidden_states_170 => add_196, erf_24, mul_171, mul_172, mul_173
# hidden_states_172 => add_197, add_198, mul_174, mul_175, rsqrt_49, sub_74, var_mean_49
triton_per_fused_gelu_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 1024, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 1024.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-12
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3uglwhxvy6jmvh4jrn4ikogtuw2zbp7rp42fe7ouo6klbowetz.py
# Source Nodes: [lm_loss], Original ATen: [aten._log_softmax]
# lm_loss => amax_24, exp_24, sub_75, sum_25
triton_red_fused__log_softmax_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 511
    rnumel = 29056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (29056*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (29056*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotzbzxga4qk5q4wyuudmt56wxeaas2kumqsunvodmaida6uh2lk.py
# Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
# lm_loss => convert_element_type, div_48, full_default_3, ne_1, ne_2, neg, sum_26, sum_27, where_1
triton_per_fused_nll_loss_forward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 511
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (1 + r0), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 29056
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 29056), "index out of bounds: 0 <= tmp7 < 29056")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (29056*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp12 = tl.log(tmp11)
    tmp13 = tmp10 - tmp12
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp2.to(tl.int64)
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp20 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp27, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1 = args
    args.clear()
    assert_size_stride(arg0_1, (29056, 1024), (1024, 1))
    assert_size_stride(arg1_1, (2, 1024), (1024, 1))
    assert_size_stride(arg2_1, (512, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg16_1, (4096, ), (1, ))
    assert_size_stride(arg17_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (1024, ), (1, ))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg32_1, (4096, ), (1, ))
    assert_size_stride(arg33_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg48_1, (4096, ), (1, ))
    assert_size_stride(arg49_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (1024, ), (1, ))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg64_1, (4096, ), (1, ))
    assert_size_stride(arg65_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg80_1, (4096, ), (1, ))
    assert_size_stride(arg81_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg96_1, (4096, ), (1, ))
    assert_size_stride(arg97_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg112_1, (4096, ), (1, ))
    assert_size_stride(arg113_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg128_1, (4096, ), (1, ))
    assert_size_stride(arg129_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg144_1, (4096, ), (1, ))
    assert_size_stride(arg145_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg160_1, (4096, ), (1, ))
    assert_size_stride(arg161_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg176_1, (4096, ), (1, ))
    assert_size_stride(arg177_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (1024, ), (1, ))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg192_1, (4096, ), (1, ))
    assert_size_stride(arg193_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg204_1, (1024, ), (1, ))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg208_1, (4096, ), (1, ))
    assert_size_stride(arg209_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg210_1, (1024, ), (1, ))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (1024, ), (1, ))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg224_1, (4096, ), (1, ))
    assert_size_stride(arg225_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (1024, ), (1, ))
    assert_size_stride(arg229_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg230_1, (1024, ), (1, ))
    assert_size_stride(arg231_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg232_1, (1024, ), (1, ))
    assert_size_stride(arg233_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (1024, ), (1, ))
    assert_size_stride(arg238_1, (1024, ), (1, ))
    assert_size_stride(arg239_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg240_1, (4096, ), (1, ))
    assert_size_stride(arg241_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg248_1, (1024, ), (1, ))
    assert_size_stride(arg249_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg252_1, (1024, ), (1, ))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg256_1, (4096, ), (1, ))
    assert_size_stride(arg257_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (1024, ), (1, ))
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg268_1, (1024, ), (1, ))
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg272_1, (4096, ), (1, ))
    assert_size_stride(arg273_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (1024, ), (1, ))
    assert_size_stride(arg276_1, (1024, ), (1, ))
    assert_size_stride(arg277_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg278_1, (1024, ), (1, ))
    assert_size_stride(arg279_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg284_1, (1024, ), (1, ))
    assert_size_stride(arg285_1, (1024, ), (1, ))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg288_1, (4096, ), (1, ))
    assert_size_stride(arg289_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (1024, ), (1, ))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg298_1, (1024, ), (1, ))
    assert_size_stride(arg299_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg300_1, (1024, ), (1, ))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg304_1, (4096, ), (1, ))
    assert_size_stride(arg305_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg320_1, (4096, ), (1, ))
    assert_size_stride(arg321_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg322_1, (1024, ), (1, ))
    assert_size_stride(arg323_1, (1024, ), (1, ))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg330_1, (1024, ), (1, ))
    assert_size_stride(arg331_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg336_1, (4096, ), (1, ))
    assert_size_stride(arg337_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg338_1, (1024, ), (1, ))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg352_1, (4096, ), (1, ))
    assert_size_stride(arg353_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (1024, ), (1, ))
    assert_size_stride(arg356_1, (1024, ), (1, ))
    assert_size_stride(arg357_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg358_1, (1024, ), (1, ))
    assert_size_stride(arg359_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg364_1, (1024, ), (1, ))
    assert_size_stride(arg365_1, (1024, ), (1, ))
    assert_size_stride(arg366_1, (1024, ), (1, ))
    assert_size_stride(arg367_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg368_1, (4096, ), (1, ))
    assert_size_stride(arg369_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg370_1, (1024, ), (1, ))
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg376_1, (1024, ), (1, ))
    assert_size_stride(arg377_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (1024, ), (1, ))
    assert_size_stride(arg382_1, (1024, ), (1, ))
    assert_size_stride(arg383_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg384_1, (4096, ), (1, ))
    assert_size_stride(arg385_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg386_1, (1024, ), (1, ))
    assert_size_stride(arg387_1, (1024, ), (1, ))
    assert_size_stride(arg388_1, (1024, ), (1, ))
    assert_size_stride(arg389_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg390_1, (1024, ), (1, ))
    assert_size_stride(arg391_1, (1024, ), (1, ))
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (29056, 1024), (1024, 1))
    assert_size_stride(arg394_1, (29056, ), (1, ))
    assert_size_stride(arg395_1, (1, 512), (512, 1))
    assert_size_stride(arg396_1, (1, 512), (512, 1))
    assert_size_stride(arg397_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, inputs_embeds, ln_outputs, position_embeddings, token_type_embeddings, token_type_ids], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.zeros]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_zeros_0.run(arg397_1, arg0_1, arg1_1, arg395_1, arg2_1, arg3_1, arg4_1, buf0, buf4, 512, 1024, grid=grid(512), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg395_1
        del arg397_1
        del arg3_1
        del arg4_1
        buf5 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg5_1, (1024, 1024), (1, 1024), 0), out=buf5)
        del arg5_1
        buf6 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), out=buf6)
        del arg7_1
        buf7 = empty((512, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), out=buf7)
        del arg9_1
        buf8 = reinterpret_tensor(buf4, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf5, arg6_1, buf8, 524288, grid=grid(524288), stream=stream0)
        del arg6_1
        buf9 = reinterpret_tensor(buf5, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf6, arg8_1, buf9, 524288, grid=grid(524288), stream=stream0)
        del arg8_1
        buf10 = reinterpret_tensor(buf6, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf7, arg10_1, buf10, 524288, grid=grid(524288), stream=stream0)
        del arg10_1
        del buf7
        # Source Nodes: [], Original ATen: []
        buf11 = aten._scaled_dot_product_efficient_attention(buf8, buf9, buf10, None, False, scale=0.125)
        buf12 = buf11[0]
        del buf11
        buf16 = reinterpret_tensor(buf9, (512, 1024), (1024, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg11_1, (1024, 1024), (1, 1024), 0), out=buf16)
        del arg11_1
        buf20 = reinterpret_tensor(buf12, (1, 512, 1024), (524288, 1024, 1), 0); del buf12  # reuse
        # Source Nodes: [attention_output, ln_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf0, buf16, arg12_1, arg13_1, arg14_1, buf20, 512, 1024, grid=grid(512), stream=stream0)
        del arg13_1
        del arg14_1
        buf21 = empty((512, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg15_1, (1024, 4096), (1, 1024), 0), out=buf21)
        del arg15_1
        buf22 = reinterpret_tensor(buf21, (1, 512, 4096), (2097152, 4096, 1), 0); del buf21  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf22, arg16_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg16_1
        buf23 = reinterpret_tensor(buf20, (512, 1024), (1024, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 1024), (1, 4096), 0), out=buf23)
        del arg17_1
        buf24 = reinterpret_tensor(buf23, (1, 512, 1024), (524288, 1024, 1), 0); del buf23  # reuse
        buf28 = reinterpret_tensor(buf8, (1, 512, 1024), (524288, 1024, 1), 0); del buf8  # reuse
        # Source Nodes: [attention_output, hidden_states_6, ln_outputs_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf24, buf0, buf16, arg12_1, arg18_1, arg19_1, arg20_1, buf28, 512, 1024, grid=grid(512), stream=stream0)
        del arg12_1
        del arg18_1
        del arg19_1
        del arg20_1
        buf29 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg21_1, (1024, 1024), (1, 1024), 0), out=buf29)
        del arg21_1
        buf30 = reinterpret_tensor(buf0, (512, 1024), (1024, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), out=buf30)
        del arg23_1
        buf31 = reinterpret_tensor(buf10, (512, 1024), (1024, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), out=buf31)
        del arg25_1
        buf32 = reinterpret_tensor(buf28, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf29, arg22_1, buf32, 524288, grid=grid(524288), stream=stream0)
        del arg22_1
        buf33 = reinterpret_tensor(buf29, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf30, arg24_1, buf33, 524288, grid=grid(524288), stream=stream0)
        del arg24_1
        buf34 = reinterpret_tensor(buf30, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf31, arg26_1, buf34, 524288, grid=grid(524288), stream=stream0)
        del arg26_1
        del buf31
        # Source Nodes: [], Original ATen: []
        buf35 = aten._scaled_dot_product_efficient_attention(buf32, buf33, buf34, None, False, scale=0.125)
        buf36 = buf35[0]
        del buf35
        buf40 = reinterpret_tensor(buf34, (512, 1024), (1024, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 1024), (1, 1024), 0), out=buf40)
        del arg27_1
        buf44 = reinterpret_tensor(buf36, (1, 512, 1024), (524288, 1024, 1), 0); del buf36  # reuse
        # Source Nodes: [attention_output_2, ln_output_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf24, buf40, arg28_1, arg29_1, arg30_1, buf44, 512, 1024, grid=grid(512), stream=stream0)
        del arg29_1
        del arg30_1
        buf45 = reinterpret_tensor(buf22, (512, 4096), (4096, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg31_1, (1024, 4096), (1, 1024), 0), out=buf45)
        del arg31_1
        buf46 = reinterpret_tensor(buf45, (1, 512, 4096), (2097152, 4096, 1), 0); del buf45  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf46, arg32_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg32_1
        buf47 = reinterpret_tensor(buf44, (512, 1024), (1024, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg33_1, (4096, 1024), (1, 4096), 0), out=buf47)
        del arg33_1
        buf48 = reinterpret_tensor(buf47, (1, 512, 1024), (524288, 1024, 1), 0); del buf47  # reuse
        buf52 = reinterpret_tensor(buf33, (1, 512, 1024), (524288, 1024, 1), 0); del buf33  # reuse
        # Source Nodes: [attention_output_2, hidden_states_13, ln_outputs_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf48, buf24, buf40, arg28_1, arg34_1, arg35_1, arg36_1, buf52, 512, 1024, grid=grid(512), stream=stream0)
        del arg28_1
        del arg34_1
        del arg35_1
        del arg36_1
        buf53 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg37_1, (1024, 1024), (1, 1024), 0), out=buf53)
        del arg37_1
        buf54 = reinterpret_tensor(buf24, (512, 1024), (1024, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), out=buf54)
        del arg39_1
        buf55 = reinterpret_tensor(buf32, (512, 1024), (1024, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), out=buf55)
        del arg41_1
        buf56 = reinterpret_tensor(buf52, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf53, arg38_1, buf56, 524288, grid=grid(524288), stream=stream0)
        del arg38_1
        buf57 = reinterpret_tensor(buf53, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf54, arg40_1, buf57, 524288, grid=grid(524288), stream=stream0)
        del arg40_1
        buf58 = reinterpret_tensor(buf54, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf55, arg42_1, buf58, 524288, grid=grid(524288), stream=stream0)
        del arg42_1
        del buf55
        # Source Nodes: [], Original ATen: []
        buf59 = aten._scaled_dot_product_efficient_attention(buf56, buf57, buf58, None, False, scale=0.125)
        buf60 = buf59[0]
        del buf59
        buf64 = reinterpret_tensor(buf58, (512, 1024), (1024, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 1024), (1, 1024), 0), out=buf64)
        del arg43_1
        buf68 = reinterpret_tensor(buf60, (1, 512, 1024), (524288, 1024, 1), 0); del buf60  # reuse
        # Source Nodes: [attention_output_4, ln_output_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf48, buf64, arg44_1, arg45_1, arg46_1, buf68, 512, 1024, grid=grid(512), stream=stream0)
        del arg45_1
        del arg46_1
        buf69 = reinterpret_tensor(buf46, (512, 4096), (4096, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg47_1, (1024, 4096), (1, 1024), 0), out=buf69)
        del arg47_1
        buf70 = reinterpret_tensor(buf69, (1, 512, 4096), (2097152, 4096, 1), 0); del buf69  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf70, arg48_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg48_1
        buf71 = reinterpret_tensor(buf68, (512, 1024), (1024, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg49_1, (4096, 1024), (1, 4096), 0), out=buf71)
        del arg49_1
        buf72 = reinterpret_tensor(buf71, (1, 512, 1024), (524288, 1024, 1), 0); del buf71  # reuse
        buf76 = reinterpret_tensor(buf57, (1, 512, 1024), (524288, 1024, 1), 0); del buf57  # reuse
        # Source Nodes: [attention_output_4, hidden_states_20, ln_outputs_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf72, buf48, buf64, arg44_1, arg50_1, arg51_1, arg52_1, buf76, 512, 1024, grid=grid(512), stream=stream0)
        del arg44_1
        del arg50_1
        del arg51_1
        del arg52_1
        buf77 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg53_1, (1024, 1024), (1, 1024), 0), out=buf77)
        del arg53_1
        buf78 = reinterpret_tensor(buf48, (512, 1024), (1024, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), out=buf78)
        del arg55_1
        buf79 = reinterpret_tensor(buf56, (512, 1024), (1024, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), out=buf79)
        del arg57_1
        buf80 = reinterpret_tensor(buf76, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf77, arg54_1, buf80, 524288, grid=grid(524288), stream=stream0)
        del arg54_1
        buf81 = reinterpret_tensor(buf77, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf78, arg56_1, buf81, 524288, grid=grid(524288), stream=stream0)
        del arg56_1
        buf82 = reinterpret_tensor(buf78, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf79, arg58_1, buf82, 524288, grid=grid(524288), stream=stream0)
        del arg58_1
        del buf79
        # Source Nodes: [], Original ATen: []
        buf83 = aten._scaled_dot_product_efficient_attention(buf80, buf81, buf82, None, False, scale=0.125)
        buf84 = buf83[0]
        del buf83
        buf88 = reinterpret_tensor(buf82, (512, 1024), (1024, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg59_1, (1024, 1024), (1, 1024), 0), out=buf88)
        del arg59_1
        buf92 = reinterpret_tensor(buf84, (1, 512, 1024), (524288, 1024, 1), 0); del buf84  # reuse
        # Source Nodes: [attention_output_6, ln_output_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf72, buf88, arg60_1, arg61_1, arg62_1, buf92, 512, 1024, grid=grid(512), stream=stream0)
        del arg61_1
        del arg62_1
        buf93 = reinterpret_tensor(buf70, (512, 4096), (4096, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg63_1, (1024, 4096), (1, 1024), 0), out=buf93)
        del arg63_1
        buf94 = reinterpret_tensor(buf93, (1, 512, 4096), (2097152, 4096, 1), 0); del buf93  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf94, arg64_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg64_1
        buf95 = reinterpret_tensor(buf92, (512, 1024), (1024, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg65_1, (4096, 1024), (1, 4096), 0), out=buf95)
        del arg65_1
        buf96 = reinterpret_tensor(buf95, (1, 512, 1024), (524288, 1024, 1), 0); del buf95  # reuse
        buf100 = reinterpret_tensor(buf81, (1, 512, 1024), (524288, 1024, 1), 0); del buf81  # reuse
        # Source Nodes: [attention_output_6, hidden_states_27, ln_outputs_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf96, buf72, buf88, arg60_1, arg66_1, arg67_1, arg68_1, buf100, 512, 1024, grid=grid(512), stream=stream0)
        del arg60_1
        del arg66_1
        del arg67_1
        del arg68_1
        buf101 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 1024), (1, 1024), 0), out=buf101)
        del arg69_1
        buf102 = reinterpret_tensor(buf72, (512, 1024), (1024, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), out=buf102)
        del arg71_1
        buf103 = reinterpret_tensor(buf80, (512, 1024), (1024, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), out=buf103)
        del arg73_1
        buf104 = reinterpret_tensor(buf100, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf101, arg70_1, buf104, 524288, grid=grid(524288), stream=stream0)
        del arg70_1
        buf105 = reinterpret_tensor(buf101, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf102, arg72_1, buf105, 524288, grid=grid(524288), stream=stream0)
        del arg72_1
        buf106 = reinterpret_tensor(buf102, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf103, arg74_1, buf106, 524288, grid=grid(524288), stream=stream0)
        del arg74_1
        del buf103
        # Source Nodes: [], Original ATen: []
        buf107 = aten._scaled_dot_product_efficient_attention(buf104, buf105, buf106, None, False, scale=0.125)
        buf108 = buf107[0]
        del buf107
        buf112 = reinterpret_tensor(buf106, (512, 1024), (1024, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 1024), (1, 1024), 0), out=buf112)
        del arg75_1
        buf116 = reinterpret_tensor(buf108, (1, 512, 1024), (524288, 1024, 1), 0); del buf108  # reuse
        # Source Nodes: [attention_output_8, ln_output_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf96, buf112, arg76_1, arg77_1, arg78_1, buf116, 512, 1024, grid=grid(512), stream=stream0)
        del arg77_1
        del arg78_1
        buf117 = reinterpret_tensor(buf94, (512, 4096), (4096, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg79_1, (1024, 4096), (1, 1024), 0), out=buf117)
        del arg79_1
        buf118 = reinterpret_tensor(buf117, (1, 512, 4096), (2097152, 4096, 1), 0); del buf117  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf118, arg80_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg80_1
        buf119 = reinterpret_tensor(buf116, (512, 1024), (1024, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg81_1, (4096, 1024), (1, 4096), 0), out=buf119)
        del arg81_1
        buf120 = reinterpret_tensor(buf119, (1, 512, 1024), (524288, 1024, 1), 0); del buf119  # reuse
        buf124 = reinterpret_tensor(buf105, (1, 512, 1024), (524288, 1024, 1), 0); del buf105  # reuse
        # Source Nodes: [attention_output_8, hidden_states_34, ln_outputs_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf120, buf96, buf112, arg76_1, arg82_1, arg83_1, arg84_1, buf124, 512, 1024, grid=grid(512), stream=stream0)
        del arg76_1
        del arg82_1
        del arg83_1
        del arg84_1
        buf125 = reinterpret_tensor(buf96, (512, 1024), (1024, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 1024), (1, 1024), 0), out=buf125)
        del arg85_1
        buf126 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), out=buf126)
        del arg87_1
        buf127 = reinterpret_tensor(buf104, (512, 1024), (1024, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), out=buf127)
        del arg89_1
        buf128 = reinterpret_tensor(buf124, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf125, arg86_1, buf128, 524288, grid=grid(524288), stream=stream0)
        del arg86_1
        buf129 = reinterpret_tensor(buf125, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf126, arg88_1, buf129, 524288, grid=grid(524288), stream=stream0)
        del arg88_1
        buf130 = reinterpret_tensor(buf126, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf127, arg90_1, buf130, 524288, grid=grid(524288), stream=stream0)
        del arg90_1
        del buf127
        # Source Nodes: [], Original ATen: []
        buf131 = aten._scaled_dot_product_efficient_attention(buf128, buf129, buf130, None, False, scale=0.125)
        buf132 = buf131[0]
        del buf131
        buf136 = reinterpret_tensor(buf130, (512, 1024), (1024, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1024), (1, 1024), 0), out=buf136)
        del arg91_1
        buf140 = reinterpret_tensor(buf132, (1, 512, 1024), (524288, 1024, 1), 0); del buf132  # reuse
        # Source Nodes: [attention_output_10, ln_output_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf120, buf136, arg92_1, arg93_1, arg94_1, buf140, 512, 1024, grid=grid(512), stream=stream0)
        del arg93_1
        del arg94_1
        buf141 = reinterpret_tensor(buf118, (512, 4096), (4096, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg95_1, (1024, 4096), (1, 1024), 0), out=buf141)
        del arg95_1
        buf142 = reinterpret_tensor(buf141, (1, 512, 4096), (2097152, 4096, 1), 0); del buf141  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf142, arg96_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg96_1
        buf143 = reinterpret_tensor(buf140, (512, 1024), (1024, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg97_1, (4096, 1024), (1, 4096), 0), out=buf143)
        del arg97_1
        buf144 = reinterpret_tensor(buf143, (1, 512, 1024), (524288, 1024, 1), 0); del buf143  # reuse
        buf148 = reinterpret_tensor(buf129, (1, 512, 1024), (524288, 1024, 1), 0); del buf129  # reuse
        # Source Nodes: [attention_output_10, hidden_states_41, ln_outputs_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf144, buf120, buf136, arg92_1, arg98_1, arg99_1, arg100_1, buf148, 512, 1024, grid=grid(512), stream=stream0)
        del arg100_1
        del arg92_1
        del arg98_1
        del arg99_1
        buf149 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 1024), (1, 1024), 0), out=buf149)
        del arg101_1
        buf150 = reinterpret_tensor(buf120, (512, 1024), (1024, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), out=buf150)
        del arg103_1
        buf151 = reinterpret_tensor(buf128, (512, 1024), (1024, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), out=buf151)
        del arg105_1
        buf152 = reinterpret_tensor(buf148, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf149, arg102_1, buf152, 524288, grid=grid(524288), stream=stream0)
        del arg102_1
        buf153 = reinterpret_tensor(buf149, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf150, arg104_1, buf153, 524288, grid=grid(524288), stream=stream0)
        del arg104_1
        buf154 = reinterpret_tensor(buf150, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf151, arg106_1, buf154, 524288, grid=grid(524288), stream=stream0)
        del arg106_1
        del buf151
        # Source Nodes: [], Original ATen: []
        buf155 = aten._scaled_dot_product_efficient_attention(buf152, buf153, buf154, None, False, scale=0.125)
        buf156 = buf155[0]
        del buf155
        buf160 = reinterpret_tensor(buf154, (512, 1024), (1024, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg107_1, (1024, 1024), (1, 1024), 0), out=buf160)
        del arg107_1
        buf164 = reinterpret_tensor(buf156, (1, 512, 1024), (524288, 1024, 1), 0); del buf156  # reuse
        # Source Nodes: [attention_output_12, ln_output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf144, buf160, arg108_1, arg109_1, arg110_1, buf164, 512, 1024, grid=grid(512), stream=stream0)
        del arg109_1
        del arg110_1
        buf165 = reinterpret_tensor(buf142, (512, 4096), (4096, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg111_1, (1024, 4096), (1, 1024), 0), out=buf165)
        del arg111_1
        buf166 = reinterpret_tensor(buf165, (1, 512, 4096), (2097152, 4096, 1), 0); del buf165  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf166, arg112_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg112_1
        buf167 = reinterpret_tensor(buf164, (512, 1024), (1024, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg113_1, (4096, 1024), (1, 4096), 0), out=buf167)
        del arg113_1
        buf168 = reinterpret_tensor(buf167, (1, 512, 1024), (524288, 1024, 1), 0); del buf167  # reuse
        buf172 = reinterpret_tensor(buf153, (1, 512, 1024), (524288, 1024, 1), 0); del buf153  # reuse
        # Source Nodes: [attention_output_12, hidden_states_48, ln_outputs_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf168, buf144, buf160, arg108_1, arg114_1, arg115_1, arg116_1, buf172, 512, 1024, grid=grid(512), stream=stream0)
        del arg108_1
        del arg114_1
        del arg115_1
        del arg116_1
        buf173 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 1024), (1, 1024), 0), out=buf173)
        del arg117_1
        buf174 = reinterpret_tensor(buf144, (512, 1024), (1024, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), out=buf174)
        del arg119_1
        buf175 = reinterpret_tensor(buf152, (512, 1024), (1024, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), out=buf175)
        del arg121_1
        buf176 = reinterpret_tensor(buf172, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf173, arg118_1, buf176, 524288, grid=grid(524288), stream=stream0)
        del arg118_1
        buf177 = reinterpret_tensor(buf173, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf174, arg120_1, buf177, 524288, grid=grid(524288), stream=stream0)
        del arg120_1
        buf178 = reinterpret_tensor(buf174, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf175, arg122_1, buf178, 524288, grid=grid(524288), stream=stream0)
        del arg122_1
        del buf175
        # Source Nodes: [], Original ATen: []
        buf179 = aten._scaled_dot_product_efficient_attention(buf176, buf177, buf178, None, False, scale=0.125)
        buf180 = buf179[0]
        del buf179
        buf184 = reinterpret_tensor(buf178, (512, 1024), (1024, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1024), (1, 1024), 0), out=buf184)
        del arg123_1
        buf188 = reinterpret_tensor(buf180, (1, 512, 1024), (524288, 1024, 1), 0); del buf180  # reuse
        # Source Nodes: [attention_output_14, ln_output_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf168, buf184, arg124_1, arg125_1, arg126_1, buf188, 512, 1024, grid=grid(512), stream=stream0)
        del arg125_1
        del arg126_1
        buf189 = reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg127_1, (1024, 4096), (1, 1024), 0), out=buf189)
        del arg127_1
        buf190 = reinterpret_tensor(buf189, (1, 512, 4096), (2097152, 4096, 1), 0); del buf189  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf190, arg128_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg128_1
        buf191 = reinterpret_tensor(buf188, (512, 1024), (1024, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg129_1, (4096, 1024), (1, 4096), 0), out=buf191)
        del arg129_1
        buf192 = reinterpret_tensor(buf191, (1, 512, 1024), (524288, 1024, 1), 0); del buf191  # reuse
        buf196 = reinterpret_tensor(buf177, (1, 512, 1024), (524288, 1024, 1), 0); del buf177  # reuse
        # Source Nodes: [attention_output_14, hidden_states_55, ln_outputs_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf192, buf168, buf184, arg124_1, arg130_1, arg131_1, arg132_1, buf196, 512, 1024, grid=grid(512), stream=stream0)
        del arg124_1
        del arg130_1
        del arg131_1
        del arg132_1
        buf197 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 1024), (1, 1024), 0), out=buf197)
        del arg133_1
        buf198 = reinterpret_tensor(buf168, (512, 1024), (1024, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), out=buf198)
        del arg135_1
        buf199 = reinterpret_tensor(buf176, (512, 1024), (1024, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), out=buf199)
        del arg137_1
        buf200 = reinterpret_tensor(buf196, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf197, arg134_1, buf200, 524288, grid=grid(524288), stream=stream0)
        del arg134_1
        buf201 = reinterpret_tensor(buf197, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf198, arg136_1, buf201, 524288, grid=grid(524288), stream=stream0)
        del arg136_1
        buf202 = reinterpret_tensor(buf198, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf199, arg138_1, buf202, 524288, grid=grid(524288), stream=stream0)
        del arg138_1
        del buf199
        # Source Nodes: [], Original ATen: []
        buf203 = aten._scaled_dot_product_efficient_attention(buf200, buf201, buf202, None, False, scale=0.125)
        buf204 = buf203[0]
        del buf203
        buf208 = reinterpret_tensor(buf202, (512, 1024), (1024, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 1024), (1, 1024), 0), out=buf208)
        del arg139_1
        buf212 = reinterpret_tensor(buf204, (1, 512, 1024), (524288, 1024, 1), 0); del buf204  # reuse
        # Source Nodes: [attention_output_16, ln_output_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf192, buf208, arg140_1, arg141_1, arg142_1, buf212, 512, 1024, grid=grid(512), stream=stream0)
        del arg141_1
        del arg142_1
        buf213 = reinterpret_tensor(buf190, (512, 4096), (4096, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg143_1, (1024, 4096), (1, 1024), 0), out=buf213)
        del arg143_1
        buf214 = reinterpret_tensor(buf213, (1, 512, 4096), (2097152, 4096, 1), 0); del buf213  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf214, arg144_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg144_1
        buf215 = reinterpret_tensor(buf212, (512, 1024), (1024, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg145_1, (4096, 1024), (1, 4096), 0), out=buf215)
        del arg145_1
        buf216 = reinterpret_tensor(buf215, (1, 512, 1024), (524288, 1024, 1), 0); del buf215  # reuse
        buf220 = reinterpret_tensor(buf201, (1, 512, 1024), (524288, 1024, 1), 0); del buf201  # reuse
        # Source Nodes: [attention_output_16, hidden_states_62, ln_outputs_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf216, buf192, buf208, arg140_1, arg146_1, arg147_1, arg148_1, buf220, 512, 1024, grid=grid(512), stream=stream0)
        del arg140_1
        del arg146_1
        del arg147_1
        del arg148_1
        buf221 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 1024), (1, 1024), 0), out=buf221)
        del arg149_1
        buf222 = reinterpret_tensor(buf192, (512, 1024), (1024, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), out=buf222)
        del arg151_1
        buf223 = reinterpret_tensor(buf200, (512, 1024), (1024, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), out=buf223)
        del arg153_1
        buf224 = reinterpret_tensor(buf220, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf221, arg150_1, buf224, 524288, grid=grid(524288), stream=stream0)
        del arg150_1
        buf225 = reinterpret_tensor(buf221, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf222, arg152_1, buf225, 524288, grid=grid(524288), stream=stream0)
        del arg152_1
        buf226 = reinterpret_tensor(buf222, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf223, arg154_1, buf226, 524288, grid=grid(524288), stream=stream0)
        del arg154_1
        del buf223
        # Source Nodes: [], Original ATen: []
        buf227 = aten._scaled_dot_product_efficient_attention(buf224, buf225, buf226, None, False, scale=0.125)
        buf228 = buf227[0]
        del buf227
        buf232 = reinterpret_tensor(buf226, (512, 1024), (1024, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 1024), (1, 1024), 0), out=buf232)
        del arg155_1
        buf236 = reinterpret_tensor(buf228, (1, 512, 1024), (524288, 1024, 1), 0); del buf228  # reuse
        # Source Nodes: [attention_output_18, ln_output_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf216, buf232, arg156_1, arg157_1, arg158_1, buf236, 512, 1024, grid=grid(512), stream=stream0)
        del arg157_1
        del arg158_1
        buf237 = reinterpret_tensor(buf214, (512, 4096), (4096, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg159_1, (1024, 4096), (1, 1024), 0), out=buf237)
        del arg159_1
        buf238 = reinterpret_tensor(buf237, (1, 512, 4096), (2097152, 4096, 1), 0); del buf237  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf238, arg160_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg160_1
        buf239 = reinterpret_tensor(buf236, (512, 1024), (1024, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg161_1, (4096, 1024), (1, 4096), 0), out=buf239)
        del arg161_1
        buf240 = reinterpret_tensor(buf239, (1, 512, 1024), (524288, 1024, 1), 0); del buf239  # reuse
        buf244 = reinterpret_tensor(buf225, (1, 512, 1024), (524288, 1024, 1), 0); del buf225  # reuse
        # Source Nodes: [attention_output_18, hidden_states_69, ln_outputs_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf240, buf216, buf232, arg156_1, arg162_1, arg163_1, arg164_1, buf244, 512, 1024, grid=grid(512), stream=stream0)
        del arg156_1
        del arg162_1
        del arg163_1
        del arg164_1
        buf245 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg165_1, (1024, 1024), (1, 1024), 0), out=buf245)
        del arg165_1
        buf246 = reinterpret_tensor(buf216, (512, 1024), (1024, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), out=buf246)
        del arg167_1
        buf247 = reinterpret_tensor(buf224, (512, 1024), (1024, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), out=buf247)
        del arg169_1
        buf248 = reinterpret_tensor(buf244, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf245, arg166_1, buf248, 524288, grid=grid(524288), stream=stream0)
        del arg166_1
        buf249 = reinterpret_tensor(buf245, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf246, arg168_1, buf249, 524288, grid=grid(524288), stream=stream0)
        del arg168_1
        buf250 = reinterpret_tensor(buf246, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf247, arg170_1, buf250, 524288, grid=grid(524288), stream=stream0)
        del arg170_1
        del buf247
        # Source Nodes: [], Original ATen: []
        buf251 = aten._scaled_dot_product_efficient_attention(buf248, buf249, buf250, None, False, scale=0.125)
        buf252 = buf251[0]
        del buf251
        buf256 = reinterpret_tensor(buf250, (512, 1024), (1024, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf252, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1024), (1, 1024), 0), out=buf256)
        del arg171_1
        buf260 = reinterpret_tensor(buf252, (1, 512, 1024), (524288, 1024, 1), 0); del buf252  # reuse
        # Source Nodes: [attention_output_20, ln_output_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf240, buf256, arg172_1, arg173_1, arg174_1, buf260, 512, 1024, grid=grid(512), stream=stream0)
        del arg173_1
        del arg174_1
        buf261 = reinterpret_tensor(buf238, (512, 4096), (4096, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg175_1, (1024, 4096), (1, 1024), 0), out=buf261)
        del arg175_1
        buf262 = reinterpret_tensor(buf261, (1, 512, 4096), (2097152, 4096, 1), 0); del buf261  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf262, arg176_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg176_1
        buf263 = reinterpret_tensor(buf260, (512, 1024), (1024, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf262, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg177_1, (4096, 1024), (1, 4096), 0), out=buf263)
        del arg177_1
        buf264 = reinterpret_tensor(buf263, (1, 512, 1024), (524288, 1024, 1), 0); del buf263  # reuse
        buf268 = reinterpret_tensor(buf249, (1, 512, 1024), (524288, 1024, 1), 0); del buf249  # reuse
        # Source Nodes: [attention_output_20, hidden_states_76, ln_outputs_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf264, buf240, buf256, arg172_1, arg178_1, arg179_1, arg180_1, buf268, 512, 1024, grid=grid(512), stream=stream0)
        del arg172_1
        del arg178_1
        del arg179_1
        del arg180_1
        buf269 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg181_1, (1024, 1024), (1, 1024), 0), out=buf269)
        del arg181_1
        buf270 = reinterpret_tensor(buf240, (512, 1024), (1024, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), out=buf270)
        del arg183_1
        buf271 = reinterpret_tensor(buf248, (512, 1024), (1024, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), out=buf271)
        del arg185_1
        buf272 = reinterpret_tensor(buf268, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf269, arg182_1, buf272, 524288, grid=grid(524288), stream=stream0)
        del arg182_1
        buf273 = reinterpret_tensor(buf269, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf270, arg184_1, buf273, 524288, grid=grid(524288), stream=stream0)
        del arg184_1
        buf274 = reinterpret_tensor(buf270, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf271, arg186_1, buf274, 524288, grid=grid(524288), stream=stream0)
        del arg186_1
        del buf271
        # Source Nodes: [], Original ATen: []
        buf275 = aten._scaled_dot_product_efficient_attention(buf272, buf273, buf274, None, False, scale=0.125)
        buf276 = buf275[0]
        del buf275
        buf280 = reinterpret_tensor(buf274, (512, 1024), (1024, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 1024), (1, 1024), 0), out=buf280)
        del arg187_1
        buf284 = reinterpret_tensor(buf276, (1, 512, 1024), (524288, 1024, 1), 0); del buf276  # reuse
        # Source Nodes: [attention_output_22, ln_output_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf264, buf280, arg188_1, arg189_1, arg190_1, buf284, 512, 1024, grid=grid(512), stream=stream0)
        del arg189_1
        del arg190_1
        buf285 = reinterpret_tensor(buf262, (512, 4096), (4096, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg191_1, (1024, 4096), (1, 1024), 0), out=buf285)
        del arg191_1
        buf286 = reinterpret_tensor(buf285, (1, 512, 4096), (2097152, 4096, 1), 0); del buf285  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf286, arg192_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg192_1
        buf287 = reinterpret_tensor(buf284, (512, 1024), (1024, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg193_1, (4096, 1024), (1, 4096), 0), out=buf287)
        del arg193_1
        buf288 = reinterpret_tensor(buf287, (1, 512, 1024), (524288, 1024, 1), 0); del buf287  # reuse
        buf292 = reinterpret_tensor(buf273, (1, 512, 1024), (524288, 1024, 1), 0); del buf273  # reuse
        # Source Nodes: [attention_output_22, hidden_states_83, ln_outputs_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf288, buf264, buf280, arg188_1, arg194_1, arg195_1, arg196_1, buf292, 512, 1024, grid=grid(512), stream=stream0)
        del arg188_1
        del arg194_1
        del arg195_1
        del arg196_1
        buf293 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg197_1, (1024, 1024), (1, 1024), 0), out=buf293)
        del arg197_1
        buf294 = reinterpret_tensor(buf264, (512, 1024), (1024, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg199_1, (1024, 1024), (1, 1024), 0), out=buf294)
        del arg199_1
        buf295 = reinterpret_tensor(buf272, (512, 1024), (1024, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 1024), (1, 1024), 0), out=buf295)
        del arg201_1
        buf296 = reinterpret_tensor(buf292, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf293, arg198_1, buf296, 524288, grid=grid(524288), stream=stream0)
        del arg198_1
        buf297 = reinterpret_tensor(buf293, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf294, arg200_1, buf297, 524288, grid=grid(524288), stream=stream0)
        del arg200_1
        buf298 = reinterpret_tensor(buf294, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf295, arg202_1, buf298, 524288, grid=grid(524288), stream=stream0)
        del arg202_1
        del buf295
        # Source Nodes: [], Original ATen: []
        buf299 = aten._scaled_dot_product_efficient_attention(buf296, buf297, buf298, None, False, scale=0.125)
        buf300 = buf299[0]
        del buf299
        buf304 = reinterpret_tensor(buf298, (512, 1024), (1024, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf300, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 1024), (1, 1024), 0), out=buf304)
        del arg203_1
        buf308 = reinterpret_tensor(buf300, (1, 512, 1024), (524288, 1024, 1), 0); del buf300  # reuse
        # Source Nodes: [attention_output_24, ln_output_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf288, buf304, arg204_1, arg205_1, arg206_1, buf308, 512, 1024, grid=grid(512), stream=stream0)
        del arg205_1
        del arg206_1
        buf309 = reinterpret_tensor(buf286, (512, 4096), (4096, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg207_1, (1024, 4096), (1, 1024), 0), out=buf309)
        del arg207_1
        buf310 = reinterpret_tensor(buf309, (1, 512, 4096), (2097152, 4096, 1), 0); del buf309  # reuse
        # Source Nodes: [intermediate_output_12], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf310, arg208_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg208_1
        buf311 = reinterpret_tensor(buf308, (512, 1024), (1024, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg209_1, (4096, 1024), (1, 4096), 0), out=buf311)
        del arg209_1
        buf312 = reinterpret_tensor(buf311, (1, 512, 1024), (524288, 1024, 1), 0); del buf311  # reuse
        buf316 = reinterpret_tensor(buf297, (1, 512, 1024), (524288, 1024, 1), 0); del buf297  # reuse
        # Source Nodes: [attention_output_24, hidden_states_90, ln_outputs_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf312, buf288, buf304, arg204_1, arg210_1, arg211_1, arg212_1, buf316, 512, 1024, grid=grid(512), stream=stream0)
        del arg204_1
        del arg210_1
        del arg211_1
        del arg212_1
        buf317 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf316, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg213_1, (1024, 1024), (1, 1024), 0), out=buf317)
        del arg213_1
        buf318 = reinterpret_tensor(buf288, (512, 1024), (1024, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf316, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg215_1, (1024, 1024), (1, 1024), 0), out=buf318)
        del arg215_1
        buf319 = reinterpret_tensor(buf296, (512, 1024), (1024, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf316, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg217_1, (1024, 1024), (1, 1024), 0), out=buf319)
        del arg217_1
        buf320 = reinterpret_tensor(buf316, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf317, arg214_1, buf320, 524288, grid=grid(524288), stream=stream0)
        del arg214_1
        buf321 = reinterpret_tensor(buf317, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf318, arg216_1, buf321, 524288, grid=grid(524288), stream=stream0)
        del arg216_1
        buf322 = reinterpret_tensor(buf318, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf319, arg218_1, buf322, 524288, grid=grid(524288), stream=stream0)
        del arg218_1
        del buf319
        # Source Nodes: [], Original ATen: []
        buf323 = aten._scaled_dot_product_efficient_attention(buf320, buf321, buf322, None, False, scale=0.125)
        buf324 = buf323[0]
        del buf323
        buf328 = reinterpret_tensor(buf322, (512, 1024), (1024, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg219_1, (1024, 1024), (1, 1024), 0), out=buf328)
        del arg219_1
        buf332 = reinterpret_tensor(buf324, (1, 512, 1024), (524288, 1024, 1), 0); del buf324  # reuse
        # Source Nodes: [attention_output_26, ln_output_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf312, buf328, arg220_1, arg221_1, arg222_1, buf332, 512, 1024, grid=grid(512), stream=stream0)
        del arg221_1
        del arg222_1
        buf333 = reinterpret_tensor(buf310, (512, 4096), (4096, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg223_1, (1024, 4096), (1, 1024), 0), out=buf333)
        del arg223_1
        buf334 = reinterpret_tensor(buf333, (1, 512, 4096), (2097152, 4096, 1), 0); del buf333  # reuse
        # Source Nodes: [intermediate_output_13], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf334, arg224_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg224_1
        buf335 = reinterpret_tensor(buf332, (512, 1024), (1024, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg225_1, (4096, 1024), (1, 4096), 0), out=buf335)
        del arg225_1
        buf336 = reinterpret_tensor(buf335, (1, 512, 1024), (524288, 1024, 1), 0); del buf335  # reuse
        buf340 = reinterpret_tensor(buf321, (1, 512, 1024), (524288, 1024, 1), 0); del buf321  # reuse
        # Source Nodes: [attention_output_26, hidden_states_97, ln_outputs_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf336, buf312, buf328, arg220_1, arg226_1, arg227_1, arg228_1, buf340, 512, 1024, grid=grid(512), stream=stream0)
        del arg220_1
        del arg226_1
        del arg227_1
        del arg228_1
        buf341 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg229_1, (1024, 1024), (1, 1024), 0), out=buf341)
        del arg229_1
        buf342 = reinterpret_tensor(buf312, (512, 1024), (1024, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg231_1, (1024, 1024), (1, 1024), 0), out=buf342)
        del arg231_1
        buf343 = reinterpret_tensor(buf320, (512, 1024), (1024, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg233_1, (1024, 1024), (1, 1024), 0), out=buf343)
        del arg233_1
        buf344 = reinterpret_tensor(buf340, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf341, arg230_1, buf344, 524288, grid=grid(524288), stream=stream0)
        del arg230_1
        buf345 = reinterpret_tensor(buf341, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf342, arg232_1, buf345, 524288, grid=grid(524288), stream=stream0)
        del arg232_1
        buf346 = reinterpret_tensor(buf342, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf343, arg234_1, buf346, 524288, grid=grid(524288), stream=stream0)
        del arg234_1
        del buf343
        # Source Nodes: [], Original ATen: []
        buf347 = aten._scaled_dot_product_efficient_attention(buf344, buf345, buf346, None, False, scale=0.125)
        buf348 = buf347[0]
        del buf347
        buf352 = reinterpret_tensor(buf346, (512, 1024), (1024, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf348, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg235_1, (1024, 1024), (1, 1024), 0), out=buf352)
        del arg235_1
        buf356 = reinterpret_tensor(buf348, (1, 512, 1024), (524288, 1024, 1), 0); del buf348  # reuse
        # Source Nodes: [attention_output_28, ln_output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf336, buf352, arg236_1, arg237_1, arg238_1, buf356, 512, 1024, grid=grid(512), stream=stream0)
        del arg237_1
        del arg238_1
        buf357 = reinterpret_tensor(buf334, (512, 4096), (4096, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg239_1, (1024, 4096), (1, 1024), 0), out=buf357)
        del arg239_1
        buf358 = reinterpret_tensor(buf357, (1, 512, 4096), (2097152, 4096, 1), 0); del buf357  # reuse
        # Source Nodes: [intermediate_output_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf358, arg240_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg240_1
        buf359 = reinterpret_tensor(buf356, (512, 1024), (1024, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg241_1, (4096, 1024), (1, 4096), 0), out=buf359)
        del arg241_1
        buf360 = reinterpret_tensor(buf359, (1, 512, 1024), (524288, 1024, 1), 0); del buf359  # reuse
        buf364 = reinterpret_tensor(buf345, (1, 512, 1024), (524288, 1024, 1), 0); del buf345  # reuse
        # Source Nodes: [attention_output_28, hidden_states_104, ln_outputs_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf360, buf336, buf352, arg236_1, arg242_1, arg243_1, arg244_1, buf364, 512, 1024, grid=grid(512), stream=stream0)
        del arg236_1
        del arg242_1
        del arg243_1
        del arg244_1
        buf365 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg245_1, (1024, 1024), (1, 1024), 0), out=buf365)
        del arg245_1
        buf366 = reinterpret_tensor(buf336, (512, 1024), (1024, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg247_1, (1024, 1024), (1, 1024), 0), out=buf366)
        del arg247_1
        buf367 = reinterpret_tensor(buf344, (512, 1024), (1024, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg249_1, (1024, 1024), (1, 1024), 0), out=buf367)
        del arg249_1
        buf368 = reinterpret_tensor(buf364, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf365, arg246_1, buf368, 524288, grid=grid(524288), stream=stream0)
        del arg246_1
        buf369 = reinterpret_tensor(buf365, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf366, arg248_1, buf369, 524288, grid=grid(524288), stream=stream0)
        del arg248_1
        buf370 = reinterpret_tensor(buf366, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf367, arg250_1, buf370, 524288, grid=grid(524288), stream=stream0)
        del arg250_1
        del buf367
        # Source Nodes: [], Original ATen: []
        buf371 = aten._scaled_dot_product_efficient_attention(buf368, buf369, buf370, None, False, scale=0.125)
        buf372 = buf371[0]
        del buf371
        buf376 = reinterpret_tensor(buf370, (512, 1024), (1024, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg251_1, (1024, 1024), (1, 1024), 0), out=buf376)
        del arg251_1
        buf380 = reinterpret_tensor(buf372, (1, 512, 1024), (524288, 1024, 1), 0); del buf372  # reuse
        # Source Nodes: [attention_output_30, ln_output_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf360, buf376, arg252_1, arg253_1, arg254_1, buf380, 512, 1024, grid=grid(512), stream=stream0)
        del arg253_1
        del arg254_1
        buf381 = reinterpret_tensor(buf358, (512, 4096), (4096, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf380, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg255_1, (1024, 4096), (1, 1024), 0), out=buf381)
        del arg255_1
        buf382 = reinterpret_tensor(buf381, (1, 512, 4096), (2097152, 4096, 1), 0); del buf381  # reuse
        # Source Nodes: [intermediate_output_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf382, arg256_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg256_1
        buf383 = reinterpret_tensor(buf380, (512, 1024), (1024, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg257_1, (4096, 1024), (1, 4096), 0), out=buf383)
        del arg257_1
        buf384 = reinterpret_tensor(buf383, (1, 512, 1024), (524288, 1024, 1), 0); del buf383  # reuse
        buf388 = reinterpret_tensor(buf369, (1, 512, 1024), (524288, 1024, 1), 0); del buf369  # reuse
        # Source Nodes: [attention_output_30, hidden_states_111, ln_outputs_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf384, buf360, buf376, arg252_1, arg258_1, arg259_1, arg260_1, buf388, 512, 1024, grid=grid(512), stream=stream0)
        del arg252_1
        del arg258_1
        del arg259_1
        del arg260_1
        buf389 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg261_1, (1024, 1024), (1, 1024), 0), out=buf389)
        del arg261_1
        buf390 = reinterpret_tensor(buf360, (512, 1024), (1024, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg263_1, (1024, 1024), (1, 1024), 0), out=buf390)
        del arg263_1
        buf391 = reinterpret_tensor(buf368, (512, 1024), (1024, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg265_1, (1024, 1024), (1, 1024), 0), out=buf391)
        del arg265_1
        buf392 = reinterpret_tensor(buf388, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf389, arg262_1, buf392, 524288, grid=grid(524288), stream=stream0)
        del arg262_1
        buf393 = reinterpret_tensor(buf389, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf390, arg264_1, buf393, 524288, grid=grid(524288), stream=stream0)
        del arg264_1
        buf394 = reinterpret_tensor(buf390, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf391, arg266_1, buf394, 524288, grid=grid(524288), stream=stream0)
        del arg266_1
        del buf391
        # Source Nodes: [], Original ATen: []
        buf395 = aten._scaled_dot_product_efficient_attention(buf392, buf393, buf394, None, False, scale=0.125)
        buf396 = buf395[0]
        del buf395
        buf400 = reinterpret_tensor(buf394, (512, 1024), (1024, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf396, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg267_1, (1024, 1024), (1, 1024), 0), out=buf400)
        del arg267_1
        buf404 = reinterpret_tensor(buf396, (1, 512, 1024), (524288, 1024, 1), 0); del buf396  # reuse
        # Source Nodes: [attention_output_32, ln_output_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf384, buf400, arg268_1, arg269_1, arg270_1, buf404, 512, 1024, grid=grid(512), stream=stream0)
        del arg269_1
        del arg270_1
        buf405 = reinterpret_tensor(buf382, (512, 4096), (4096, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg271_1, (1024, 4096), (1, 1024), 0), out=buf405)
        del arg271_1
        buf406 = reinterpret_tensor(buf405, (1, 512, 4096), (2097152, 4096, 1), 0); del buf405  # reuse
        # Source Nodes: [intermediate_output_16], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf406, arg272_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg272_1
        buf407 = reinterpret_tensor(buf404, (512, 1024), (1024, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf406, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg273_1, (4096, 1024), (1, 4096), 0), out=buf407)
        del arg273_1
        buf408 = reinterpret_tensor(buf407, (1, 512, 1024), (524288, 1024, 1), 0); del buf407  # reuse
        buf412 = reinterpret_tensor(buf393, (1, 512, 1024), (524288, 1024, 1), 0); del buf393  # reuse
        # Source Nodes: [attention_output_32, hidden_states_118, ln_outputs_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf408, buf384, buf400, arg268_1, arg274_1, arg275_1, arg276_1, buf412, 512, 1024, grid=grid(512), stream=stream0)
        del arg268_1
        del arg274_1
        del arg275_1
        del arg276_1
        buf413 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg277_1, (1024, 1024), (1, 1024), 0), out=buf413)
        del arg277_1
        buf414 = reinterpret_tensor(buf384, (512, 1024), (1024, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg279_1, (1024, 1024), (1, 1024), 0), out=buf414)
        del arg279_1
        buf415 = reinterpret_tensor(buf392, (512, 1024), (1024, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg281_1, (1024, 1024), (1, 1024), 0), out=buf415)
        del arg281_1
        buf416 = reinterpret_tensor(buf412, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf413, arg278_1, buf416, 524288, grid=grid(524288), stream=stream0)
        del arg278_1
        buf417 = reinterpret_tensor(buf413, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf414, arg280_1, buf417, 524288, grid=grid(524288), stream=stream0)
        del arg280_1
        buf418 = reinterpret_tensor(buf414, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf415, arg282_1, buf418, 524288, grid=grid(524288), stream=stream0)
        del arg282_1
        del buf415
        # Source Nodes: [], Original ATen: []
        buf419 = aten._scaled_dot_product_efficient_attention(buf416, buf417, buf418, None, False, scale=0.125)
        buf420 = buf419[0]
        del buf419
        buf424 = reinterpret_tensor(buf418, (512, 1024), (1024, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf420, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 1024), (1, 1024), 0), out=buf424)
        del arg283_1
        buf428 = reinterpret_tensor(buf420, (1, 512, 1024), (524288, 1024, 1), 0); del buf420  # reuse
        # Source Nodes: [attention_output_34, ln_output_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf408, buf424, arg284_1, arg285_1, arg286_1, buf428, 512, 1024, grid=grid(512), stream=stream0)
        del arg285_1
        del arg286_1
        buf429 = reinterpret_tensor(buf406, (512, 4096), (4096, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg287_1, (1024, 4096), (1, 1024), 0), out=buf429)
        del arg287_1
        buf430 = reinterpret_tensor(buf429, (1, 512, 4096), (2097152, 4096, 1), 0); del buf429  # reuse
        # Source Nodes: [intermediate_output_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf430, arg288_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg288_1
        buf431 = reinterpret_tensor(buf428, (512, 1024), (1024, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf430, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg289_1, (4096, 1024), (1, 4096), 0), out=buf431)
        del arg289_1
        buf432 = reinterpret_tensor(buf431, (1, 512, 1024), (524288, 1024, 1), 0); del buf431  # reuse
        buf436 = reinterpret_tensor(buf417, (1, 512, 1024), (524288, 1024, 1), 0); del buf417  # reuse
        # Source Nodes: [attention_output_34, hidden_states_125, ln_outputs_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf432, buf408, buf424, arg284_1, arg290_1, arg291_1, arg292_1, buf436, 512, 1024, grid=grid(512), stream=stream0)
        del arg284_1
        del arg290_1
        del arg291_1
        del arg292_1
        buf437 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg293_1, (1024, 1024), (1, 1024), 0), out=buf437)
        del arg293_1
        buf438 = reinterpret_tensor(buf408, (512, 1024), (1024, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg295_1, (1024, 1024), (1, 1024), 0), out=buf438)
        del arg295_1
        buf439 = reinterpret_tensor(buf416, (512, 1024), (1024, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg297_1, (1024, 1024), (1, 1024), 0), out=buf439)
        del arg297_1
        buf440 = reinterpret_tensor(buf436, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf437, arg294_1, buf440, 524288, grid=grid(524288), stream=stream0)
        del arg294_1
        buf441 = reinterpret_tensor(buf437, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf438, arg296_1, buf441, 524288, grid=grid(524288), stream=stream0)
        del arg296_1
        buf442 = reinterpret_tensor(buf438, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf439, arg298_1, buf442, 524288, grid=grid(524288), stream=stream0)
        del arg298_1
        del buf439
        # Source Nodes: [], Original ATen: []
        buf443 = aten._scaled_dot_product_efficient_attention(buf440, buf441, buf442, None, False, scale=0.125)
        buf444 = buf443[0]
        del buf443
        buf448 = reinterpret_tensor(buf442, (512, 1024), (1024, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg299_1, (1024, 1024), (1, 1024), 0), out=buf448)
        del arg299_1
        buf452 = reinterpret_tensor(buf444, (1, 512, 1024), (524288, 1024, 1), 0); del buf444  # reuse
        # Source Nodes: [attention_output_36, ln_output_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf432, buf448, arg300_1, arg301_1, arg302_1, buf452, 512, 1024, grid=grid(512), stream=stream0)
        del arg301_1
        del arg302_1
        buf453 = reinterpret_tensor(buf430, (512, 4096), (4096, 1), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf452, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg303_1, (1024, 4096), (1, 1024), 0), out=buf453)
        del arg303_1
        buf454 = reinterpret_tensor(buf453, (1, 512, 4096), (2097152, 4096, 1), 0); del buf453  # reuse
        # Source Nodes: [intermediate_output_18], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf454, arg304_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg304_1
        buf455 = reinterpret_tensor(buf452, (512, 1024), (1024, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf454, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg305_1, (4096, 1024), (1, 4096), 0), out=buf455)
        del arg305_1
        buf456 = reinterpret_tensor(buf455, (1, 512, 1024), (524288, 1024, 1), 0); del buf455  # reuse
        buf460 = reinterpret_tensor(buf441, (1, 512, 1024), (524288, 1024, 1), 0); del buf441  # reuse
        # Source Nodes: [attention_output_36, hidden_states_132, ln_outputs_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf456, buf432, buf448, arg300_1, arg306_1, arg307_1, arg308_1, buf460, 512, 1024, grid=grid(512), stream=stream0)
        del arg300_1
        del arg306_1
        del arg307_1
        del arg308_1
        buf461 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 1024), (1, 1024), 0), out=buf461)
        del arg309_1
        buf462 = reinterpret_tensor(buf432, (512, 1024), (1024, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg311_1, (1024, 1024), (1, 1024), 0), out=buf462)
        del arg311_1
        buf463 = reinterpret_tensor(buf440, (512, 1024), (1024, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg313_1, (1024, 1024), (1, 1024), 0), out=buf463)
        del arg313_1
        buf464 = reinterpret_tensor(buf460, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf461, arg310_1, buf464, 524288, grid=grid(524288), stream=stream0)
        del arg310_1
        buf465 = reinterpret_tensor(buf461, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf462, arg312_1, buf465, 524288, grid=grid(524288), stream=stream0)
        del arg312_1
        buf466 = reinterpret_tensor(buf462, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf463, arg314_1, buf466, 524288, grid=grid(524288), stream=stream0)
        del arg314_1
        del buf463
        # Source Nodes: [], Original ATen: []
        buf467 = aten._scaled_dot_product_efficient_attention(buf464, buf465, buf466, None, False, scale=0.125)
        buf468 = buf467[0]
        del buf467
        buf472 = reinterpret_tensor(buf466, (512, 1024), (1024, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf468, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 1024), (1, 1024), 0), out=buf472)
        del arg315_1
        buf476 = reinterpret_tensor(buf468, (1, 512, 1024), (524288, 1024, 1), 0); del buf468  # reuse
        # Source Nodes: [attention_output_38, ln_output_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf456, buf472, arg316_1, arg317_1, arg318_1, buf476, 512, 1024, grid=grid(512), stream=stream0)
        del arg317_1
        del arg318_1
        buf477 = reinterpret_tensor(buf454, (512, 4096), (4096, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf476, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg319_1, (1024, 4096), (1, 1024), 0), out=buf477)
        del arg319_1
        buf478 = reinterpret_tensor(buf477, (1, 512, 4096), (2097152, 4096, 1), 0); del buf477  # reuse
        # Source Nodes: [intermediate_output_19], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf478, arg320_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg320_1
        buf479 = reinterpret_tensor(buf476, (512, 1024), (1024, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg321_1, (4096, 1024), (1, 4096), 0), out=buf479)
        del arg321_1
        buf480 = reinterpret_tensor(buf479, (1, 512, 1024), (524288, 1024, 1), 0); del buf479  # reuse
        buf484 = reinterpret_tensor(buf465, (1, 512, 1024), (524288, 1024, 1), 0); del buf465  # reuse
        # Source Nodes: [attention_output_38, hidden_states_139, ln_outputs_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf480, buf456, buf472, arg316_1, arg322_1, arg323_1, arg324_1, buf484, 512, 1024, grid=grid(512), stream=stream0)
        del arg316_1
        del arg322_1
        del arg323_1
        del arg324_1
        buf485 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf484, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg325_1, (1024, 1024), (1, 1024), 0), out=buf485)
        del arg325_1
        buf486 = reinterpret_tensor(buf456, (512, 1024), (1024, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf484, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg327_1, (1024, 1024), (1, 1024), 0), out=buf486)
        del arg327_1
        buf487 = reinterpret_tensor(buf464, (512, 1024), (1024, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf484, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg329_1, (1024, 1024), (1, 1024), 0), out=buf487)
        del arg329_1
        buf488 = reinterpret_tensor(buf484, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf485, arg326_1, buf488, 524288, grid=grid(524288), stream=stream0)
        del arg326_1
        buf489 = reinterpret_tensor(buf485, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf486, arg328_1, buf489, 524288, grid=grid(524288), stream=stream0)
        del arg328_1
        buf490 = reinterpret_tensor(buf486, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf487, arg330_1, buf490, 524288, grid=grid(524288), stream=stream0)
        del arg330_1
        del buf487
        # Source Nodes: [], Original ATen: []
        buf491 = aten._scaled_dot_product_efficient_attention(buf488, buf489, buf490, None, False, scale=0.125)
        buf492 = buf491[0]
        del buf491
        buf496 = reinterpret_tensor(buf490, (512, 1024), (1024, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf492, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg331_1, (1024, 1024), (1, 1024), 0), out=buf496)
        del arg331_1
        buf500 = reinterpret_tensor(buf492, (1, 512, 1024), (524288, 1024, 1), 0); del buf492  # reuse
        # Source Nodes: [attention_output_40, ln_output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf480, buf496, arg332_1, arg333_1, arg334_1, buf500, 512, 1024, grid=grid(512), stream=stream0)
        del arg333_1
        del arg334_1
        buf501 = reinterpret_tensor(buf478, (512, 4096), (4096, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg335_1, (1024, 4096), (1, 1024), 0), out=buf501)
        del arg335_1
        buf502 = reinterpret_tensor(buf501, (1, 512, 4096), (2097152, 4096, 1), 0); del buf501  # reuse
        # Source Nodes: [intermediate_output_20], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf502, arg336_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg336_1
        buf503 = reinterpret_tensor(buf500, (512, 1024), (1024, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf502, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg337_1, (4096, 1024), (1, 4096), 0), out=buf503)
        del arg337_1
        buf504 = reinterpret_tensor(buf503, (1, 512, 1024), (524288, 1024, 1), 0); del buf503  # reuse
        buf508 = reinterpret_tensor(buf489, (1, 512, 1024), (524288, 1024, 1), 0); del buf489  # reuse
        # Source Nodes: [attention_output_40, hidden_states_146, ln_outputs_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf504, buf480, buf496, arg332_1, arg338_1, arg339_1, arg340_1, buf508, 512, 1024, grid=grid(512), stream=stream0)
        del arg332_1
        del arg338_1
        del arg339_1
        del arg340_1
        buf509 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf508, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg341_1, (1024, 1024), (1, 1024), 0), out=buf509)
        del arg341_1
        buf510 = reinterpret_tensor(buf480, (512, 1024), (1024, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf508, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 1024), (1, 1024), 0), out=buf510)
        del arg343_1
        buf511 = reinterpret_tensor(buf488, (512, 1024), (1024, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf508, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg345_1, (1024, 1024), (1, 1024), 0), out=buf511)
        del arg345_1
        buf512 = reinterpret_tensor(buf508, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf509, arg342_1, buf512, 524288, grid=grid(524288), stream=stream0)
        del arg342_1
        buf513 = reinterpret_tensor(buf509, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf510, arg344_1, buf513, 524288, grid=grid(524288), stream=stream0)
        del arg344_1
        buf514 = reinterpret_tensor(buf510, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf511, arg346_1, buf514, 524288, grid=grid(524288), stream=stream0)
        del arg346_1
        del buf511
        # Source Nodes: [], Original ATen: []
        buf515 = aten._scaled_dot_product_efficient_attention(buf512, buf513, buf514, None, False, scale=0.125)
        buf516 = buf515[0]
        del buf515
        buf520 = reinterpret_tensor(buf514, (512, 1024), (1024, 1), 0); del buf514  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf516, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg347_1, (1024, 1024), (1, 1024), 0), out=buf520)
        del arg347_1
        buf524 = reinterpret_tensor(buf516, (1, 512, 1024), (524288, 1024, 1), 0); del buf516  # reuse
        # Source Nodes: [attention_output_42, ln_output_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf504, buf520, arg348_1, arg349_1, arg350_1, buf524, 512, 1024, grid=grid(512), stream=stream0)
        del arg349_1
        del arg350_1
        buf525 = reinterpret_tensor(buf502, (512, 4096), (4096, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg351_1, (1024, 4096), (1, 1024), 0), out=buf525)
        del arg351_1
        buf526 = reinterpret_tensor(buf525, (1, 512, 4096), (2097152, 4096, 1), 0); del buf525  # reuse
        # Source Nodes: [intermediate_output_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf526, arg352_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg352_1
        buf527 = reinterpret_tensor(buf524, (512, 1024), (1024, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf526, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg353_1, (4096, 1024), (1, 4096), 0), out=buf527)
        del arg353_1
        buf528 = reinterpret_tensor(buf527, (1, 512, 1024), (524288, 1024, 1), 0); del buf527  # reuse
        buf532 = reinterpret_tensor(buf513, (1, 512, 1024), (524288, 1024, 1), 0); del buf513  # reuse
        # Source Nodes: [attention_output_42, hidden_states_153, ln_outputs_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf528, buf504, buf520, arg348_1, arg354_1, arg355_1, arg356_1, buf532, 512, 1024, grid=grid(512), stream=stream0)
        del arg348_1
        del arg354_1
        del arg355_1
        del arg356_1
        buf533 = buf520; del buf520  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf532, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg357_1, (1024, 1024), (1, 1024), 0), out=buf533)
        del arg357_1
        buf534 = reinterpret_tensor(buf504, (512, 1024), (1024, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf532, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg359_1, (1024, 1024), (1, 1024), 0), out=buf534)
        del arg359_1
        buf535 = reinterpret_tensor(buf512, (512, 1024), (1024, 1), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf532, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg361_1, (1024, 1024), (1, 1024), 0), out=buf535)
        del arg361_1
        buf536 = reinterpret_tensor(buf532, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf533, arg358_1, buf536, 524288, grid=grid(524288), stream=stream0)
        del arg358_1
        buf537 = reinterpret_tensor(buf533, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf533  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf534, arg360_1, buf537, 524288, grid=grid(524288), stream=stream0)
        del arg360_1
        buf538 = reinterpret_tensor(buf534, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf534  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf535, arg362_1, buf538, 524288, grid=grid(524288), stream=stream0)
        del arg362_1
        del buf535
        # Source Nodes: [], Original ATen: []
        buf539 = aten._scaled_dot_product_efficient_attention(buf536, buf537, buf538, None, False, scale=0.125)
        buf540 = buf539[0]
        del buf539
        buf544 = reinterpret_tensor(buf538, (512, 1024), (1024, 1), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg363_1, (1024, 1024), (1, 1024), 0), out=buf544)
        del arg363_1
        buf548 = reinterpret_tensor(buf540, (1, 512, 1024), (524288, 1024, 1), 0); del buf540  # reuse
        # Source Nodes: [attention_output_44, ln_output_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf528, buf544, arg364_1, arg365_1, arg366_1, buf548, 512, 1024, grid=grid(512), stream=stream0)
        del arg365_1
        del arg366_1
        buf549 = reinterpret_tensor(buf526, (512, 4096), (4096, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg367_1, (1024, 4096), (1, 1024), 0), out=buf549)
        del arg367_1
        buf550 = reinterpret_tensor(buf549, (1, 512, 4096), (2097152, 4096, 1), 0); del buf549  # reuse
        # Source Nodes: [intermediate_output_22], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf550, arg368_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg368_1
        buf551 = reinterpret_tensor(buf548, (512, 1024), (1024, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf550, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg369_1, (4096, 1024), (1, 4096), 0), out=buf551)
        del arg369_1
        buf552 = reinterpret_tensor(buf551, (1, 512, 1024), (524288, 1024, 1), 0); del buf551  # reuse
        buf556 = reinterpret_tensor(buf537, (1, 512, 1024), (524288, 1024, 1), 0); del buf537  # reuse
        # Source Nodes: [attention_output_44, hidden_states_160, ln_outputs_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf552, buf528, buf544, arg364_1, arg370_1, arg371_1, arg372_1, buf556, 512, 1024, grid=grid(512), stream=stream0)
        del arg364_1
        del arg370_1
        del arg371_1
        del arg372_1
        buf557 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf556, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg373_1, (1024, 1024), (1, 1024), 0), out=buf557)
        del arg373_1
        buf558 = reinterpret_tensor(buf528, (512, 1024), (1024, 1), 0); del buf528  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf556, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg375_1, (1024, 1024), (1, 1024), 0), out=buf558)
        del arg375_1
        buf559 = reinterpret_tensor(buf536, (512, 1024), (1024, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf556, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg377_1, (1024, 1024), (1, 1024), 0), out=buf559)
        del arg377_1
        buf560 = reinterpret_tensor(buf556, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf557, arg374_1, buf560, 524288, grid=grid(524288), stream=stream0)
        del arg374_1
        buf561 = reinterpret_tensor(buf557, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf558, arg376_1, buf561, 524288, grid=grid(524288), stream=stream0)
        del arg376_1
        buf562 = reinterpret_tensor(buf558, (1, 16, 512, 64), (524288, 32768, 64, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf559, arg378_1, buf562, 524288, grid=grid(524288), stream=stream0)
        del arg378_1
        del buf559
        # Source Nodes: [], Original ATen: []
        buf563 = aten._scaled_dot_product_efficient_attention(buf560, buf561, buf562, None, False, scale=0.125)
        del buf560
        buf564 = buf563[0]
        del buf563
        buf568 = reinterpret_tensor(buf562, (512, 1024), (1024, 1), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf564, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg379_1, (1024, 1024), (1, 1024), 0), out=buf568)
        del arg379_1
        buf572 = reinterpret_tensor(buf564, (1, 512, 1024), (524288, 1024, 1), 0); del buf564  # reuse
        # Source Nodes: [attention_output_46, ln_output_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_2.run(buf552, buf568, arg380_1, arg381_1, arg382_1, buf572, 512, 1024, grid=grid(512), stream=stream0)
        del arg381_1
        del arg382_1
        buf573 = reinterpret_tensor(buf550, (512, 4096), (4096, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf572, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg383_1, (1024, 4096), (1, 1024), 0), out=buf573)
        del arg383_1
        buf574 = reinterpret_tensor(buf573, (1, 512, 4096), (2097152, 4096, 1), 0); del buf573  # reuse
        # Source Nodes: [intermediate_output_23], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_3.run(buf574, arg384_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg384_1
        buf575 = reinterpret_tensor(buf572, (512, 1024), (1024, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf574, (512, 4096), (4096, 1), 0), reinterpret_tensor(arg385_1, (4096, 1024), (1, 4096), 0), out=buf575)
        del arg385_1
        del buf574
        buf576 = reinterpret_tensor(buf575, (1, 512, 1024), (524288, 1024, 1), 0); del buf575  # reuse
        buf580 = reinterpret_tensor(buf561, (1, 512, 1024), (524288, 1024, 1), 0); del buf561  # reuse
        # Source Nodes: [attention_output_46, hidden_states_167, sequence_output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_4.run(buf576, buf552, buf568, arg380_1, arg386_1, arg387_1, arg388_1, buf580, 512, 1024, grid=grid(512), stream=stream0)
        del arg380_1
        del arg386_1
        del arg387_1
        del arg388_1
        del buf552
        del buf568
        buf581 = reinterpret_tensor(buf576, (512, 1024), (1024, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg389_1, (1024, 1024), (1, 1024), 0), out=buf581)
        del arg389_1
        buf585 = buf580; del buf580  # reuse
        # Source Nodes: [hidden_states_170, hidden_states_172], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_5.run(buf581, arg390_1, arg391_1, arg392_1, buf585, 512, 1024, grid=grid(512), stream=stream0)
        del arg390_1
        del arg391_1
        del arg392_1
        del buf581
        buf586 = empty((512, 29056), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg394_1, reinterpret_tensor(buf585, (512, 1024), (1024, 1), 0), reinterpret_tensor(arg393_1, (1024, 29056), (1, 1024), 0), alpha=1, beta=1, out=buf586)
        del arg393_1
        del arg394_1
        del buf585
        buf587 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        buf588 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_6.run(buf586, buf587, buf588, 511, 29056, grid=grid(511), stream=stream0)
        buf589 = empty((), device='cuda', dtype=torch.float32)
        buf591 = buf589; del buf589  # reuse
        # Source Nodes: [lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_7.run(buf591, arg396_1, buf586, buf587, buf588, 1, 511, grid=grid(1), stream=stream0)
        del arg396_1
        return (buf591, reinterpret_tensor(buf586, (1, 512, 29056), (14876672, 29056, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((29056, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((29056, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((29056, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg396_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg397_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MegatronBertForCausalLM', benchmark_compiled_module)
