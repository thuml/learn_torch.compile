
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


# kernel path: /tmp/torchinductor_youkaichao/36/c36577urw5fklsfkxypiydw6x4lup4ii4z6cvsuwmq4jdrcajvzh.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# inputs_embeds => embedding
# position_embeddings => embedding_2
# token_type_embeddings => embedding_1
triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 30522
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30522")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5 + 2
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 2)) | ~xmask, "index out of bounds: 0 <= tmp8 < 2")
    tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp11 + 512
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.device_assert(((0 <= tmp14) & (tmp14 < 512)) | ~xmask, "index out of bounds: 0 <= tmp14 < 512")
    tmp15 = tl.load(in_ptr5 + (r1 + (768*tmp14)), rmask & xmask, other=0.0)
    tmp16 = tmp10 + tmp15
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
    tmp33 = tmp16 - tmp26
    tmp34 = 768.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-12
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp16, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp43, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp44, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2s3ic7nygtbrrgdj5txjepzrncknc7p3bsvfeu5pcyzyis6bmb.py
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
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zznbkpfqbzww64kqzewuirsg6csbimqfdkc5fvddd3dvgtty2l.py
# Source Nodes: [hidden_states], Original ATen: [aten.view]
# hidden_states => view_16
triton_poi_fused_view_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44svb55ep4fvrgoqnmzbdh3kunxcrfzcg2222r3vbybsksfdhxk.py
# Source Nodes: [add_2, attention_output, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_2 => add_5
# attention_output => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
# hidden_states_3 => view_18
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
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
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
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
    tmp22 = 1e-12
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


# kernel path: /tmp/torchinductor_youkaichao/gw/cgw4khgz6sdjlwotvwxxufaqlvgkqp7krfkpscx3fuaeq2p7so2q.py
# Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
# hidden_states_5 => view_20
# intermediate_output => add_8, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/fj/cfjwbp3el4wihtvvldp4bayku6dtnvfhyhbs23p7ljeofx7pvayg.py
# Source Nodes: [add_3, attention_output, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_3 => add_9
# attention_output => add_7, mul_4
# hidden_states_8 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_4, var_mean_2
# mixed_query_layer_1 => view_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
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
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
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
    tmp26 = 1e-12
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 / tmp24
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cepp2cndmaxbk7zdimedj5zzxpwcwpyreaqa2nkcbxqjzna5457u.py
# Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
# start_logits_1 => clone_12
# start_loss => amax_12, exp_12, log, sub_38, sub_39, sum_13
triton_per_fused__log_softmax_clone_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (2*r0), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/et/ceto3p2y6tkwbvqxherskymx627nkhddoqo5r2uq5t6iytpxv5xf.py
# Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
# end_logits_1 => clone_13
# end_loss => amax_13, exp_13, log_1, sub_40, sub_41, sum_16
triton_per_fused__log_softmax_clone_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (1 + (2*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (1))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.log(tmp13)
    tmp15 = tmp8 - tmp14
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(r0, [RBLOCK])), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6gg7zwfokol7qemizffgy6vifdfsxufgxulp755ftmsa6pbttb.py
# Source Nodes: [add_37, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# add_37 => add_100
# end_loss => convert_element_type_1, div_25, ne_3, neg_1, sum_17, sum_18, where_3
# end_positions => clamp_max_1, clamp_min_1
# loss => div_26
# start_loss => convert_element_type, div_24, full_default_1, full_default_2, ne, neg, sum_14, sum_15, where_1
# start_positions => clamp_max, clamp_min
triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: '*fp32', 7: '*i1', 8: '*i64', 9: '*i1', 10: '*i64', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp9 = triton_helpers.maximum(tmp8, tmp2)
    tmp10 = triton_helpers.minimum(tmp9, tmp4)
    tmp11 = tmp10 != tmp4
    tmp12 = tl.where(tmp6, tmp5, tmp2)
    tmp13 = tmp12 + 512
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 512), "index out of bounds: 0 <= tmp15 < 512")
    tmp16 = tl.load(in_ptr2 + (tmp15), None, eviction_policy='evict_last')
    tmp17 = -tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tmp6.to(tl.int64)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tl.where(tmp11, tmp10, tmp2)
    tmp24 = tmp23 + 512
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert((0 <= tmp26) & (tmp26 < 512), "index out of bounds: 0 <= tmp26 < 512")
    tmp27 = tl.load(in_ptr3 + (tmp26), None, eviction_policy='evict_last')
    tmp28 = -tmp27
    tmp29 = tl.where(tmp11, tmp28, tmp18)
    tmp30 = tmp11.to(tl.int64)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp22 + tmp32
    tmp34 = 2.0
    tmp35 = tmp33 / tmp34
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp35, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp23, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204 = args
    args.clear()
    assert_size_stride(primals_1, (30522, 768), (768, 1))
    assert_size_stride(primals_2, (2, 768), (768, 1))
    assert_size_stride(primals_3, (512, 768), (768, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, 768), (768, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, 768), (768, 1))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, 768), (768, 1))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (3072, 768), (768, 1))
    assert_size_stride(primals_17, (3072, ), (1, ))
    assert_size_stride(primals_18, (768, 3072), (3072, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, 768), (768, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, 768), (768, 1))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, 768), (768, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, 768), (768, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (3072, 768), (768, 1))
    assert_size_stride(primals_33, (3072, ), (1, ))
    assert_size_stride(primals_34, (768, 3072), (3072, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, 768), (768, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, 768), (768, 1))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, 768), (768, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, 768), (768, 1))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (3072, 768), (768, 1))
    assert_size_stride(primals_49, (3072, ), (1, ))
    assert_size_stride(primals_50, (768, 3072), (3072, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, 768), (768, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, 768), (768, 1))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, 768), (768, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, 768), (768, 1))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (3072, 768), (768, 1))
    assert_size_stride(primals_65, (3072, ), (1, ))
    assert_size_stride(primals_66, (768, 3072), (3072, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, 768), (768, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, 768), (768, 1))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, 768), (768, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, 768), (768, 1))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (3072, 768), (768, 1))
    assert_size_stride(primals_81, (3072, ), (1, ))
    assert_size_stride(primals_82, (768, 3072), (3072, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (768, 768), (768, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, 768), (768, 1))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, 768), (768, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, 768), (768, 1))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (3072, 768), (768, 1))
    assert_size_stride(primals_97, (3072, ), (1, ))
    assert_size_stride(primals_98, (768, 3072), (3072, 1))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, 768), (768, 1))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, 768), (768, 1))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, 768), (768, 1))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (3072, 768), (768, 1))
    assert_size_stride(primals_113, (3072, ), (1, ))
    assert_size_stride(primals_114, (768, 3072), (3072, 1))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, 768), (768, 1))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, 768), (768, 1))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, 768), (768, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (3072, 768), (768, 1))
    assert_size_stride(primals_129, (3072, ), (1, ))
    assert_size_stride(primals_130, (768, 3072), (3072, 1))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (768, 768), (768, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, 768), (768, 1))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, 768), (768, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, 768), (768, 1))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (3072, 768), (768, 1))
    assert_size_stride(primals_145, (3072, ), (1, ))
    assert_size_stride(primals_146, (768, 3072), (3072, 1))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (768, 768), (768, 1))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_152, (768, 768), (768, 1))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (768, 768), (768, 1))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (768, 768), (768, 1))
    assert_size_stride(primals_157, (768, ), (1, ))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_160, (3072, 768), (768, 1))
    assert_size_stride(primals_161, (3072, ), (1, ))
    assert_size_stride(primals_162, (768, 3072), (3072, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, ), (1, ))
    assert_size_stride(primals_166, (768, 768), (768, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (768, 768), (768, 1))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (768, 768), (768, 1))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (768, 768), (768, 1))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (3072, 768), (768, 1))
    assert_size_stride(primals_177, (3072, ), (1, ))
    assert_size_stride(primals_178, (768, 3072), (3072, 1))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (768, ), (1, ))
    assert_size_stride(primals_182, (768, 768), (768, 1))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (768, 768), (768, 1))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, 768), (768, 1))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (768, 768), (768, 1))
    assert_size_stride(primals_189, (768, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (3072, 768), (768, 1))
    assert_size_stride(primals_193, (3072, ), (1, ))
    assert_size_stride(primals_194, (768, 3072), (3072, 1))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (2, 768), (768, 1))
    assert_size_stride(primals_199, (2, ), (1, ))
    assert_size_stride(primals_200, (1, 512), (512, 1))
    assert_size_stride(primals_201, (1, 512), (512, 1))
    assert_size_stride(primals_202, (1, 512), (512, 1))
    assert_size_stride(primals_203, (1, ), (1, ))
    assert_size_stride(primals_204, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf432 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0.run(primals_202, primals_1, primals_200, primals_2, primals_201, primals_3, primals_4, primals_5, buf0, buf4, buf5, buf432, 512, 768, grid=grid(512), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        # Source Nodes: [embedding_output, embeddings_2], Original ATen: [aten.native_dropout, aten.native_layer_norm]
        buf6 = aten.native_dropout(buf5, 0.1, True)
        buf7 = buf6[0]
        buf8 = buf6[1]
        del buf6
        buf9 = reinterpret_tensor(buf5, (512, 768), (768, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_6, (768, 768), (1, 768), 0), out=buf9)
        buf10 = reinterpret_tensor(buf0, (512, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf10)
        buf11 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf11)
        buf12 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf9, primals_7, buf12, 393216, grid=grid(393216), stream=stream0)
        del primals_7
        buf13 = reinterpret_tensor(buf9, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf10, primals_9, buf13, 393216, grid=grid(393216), stream=stream0)
        del primals_9
        buf14 = reinterpret_tensor(buf10, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf11, primals_11, buf14, 393216, grid=grid(393216), stream=stream0)
        del primals_11
        # Source Nodes: [], Original ATen: []
        buf15 = aten._scaled_dot_product_efficient_attention(buf12, buf13, buf14, None, True, 0.1, scale=0.125)
        buf16 = buf15[0]
        buf17 = buf15[1]
        buf18 = buf15[2]
        buf19 = buf15[3]
        del buf15
        buf20 = buf11; del buf11  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf16, buf20, 393216, grid=grid(393216), stream=stream0)
        buf21 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf20, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf21)
        del primals_13
        # Source Nodes: [hidden_states_1], Original ATen: [aten.native_dropout]
        buf22 = aten.native_dropout(reinterpret_tensor(buf21, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf23 = buf22[0]
        buf24 = buf22[1]
        del buf22
        buf28 = reinterpret_tensor(buf21, (1, 512, 768), (393216, 768, 1), 0); del buf21  # reuse
        buf29 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf431 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, attention_output, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf23, buf7, primals_14, primals_15, buf28, buf29, buf431, 512, 768, grid=grid(512), stream=stream0)
        buf30 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, buf29, reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf30)
        del primals_17
        buf31 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf30, buf31, 1572864, grid=grid(1572864), stream=stream0)
        buf32 = reinterpret_tensor(buf23, (512, 768), (768, 1), 0); del buf23  # reuse
        # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf31, reinterpret_tensor(primals_18, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf32)
        del primals_19
        # Source Nodes: [hidden_states_6], Original ATen: [aten.native_dropout]
        buf33 = aten.native_dropout(reinterpret_tensor(buf32, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf39 = reinterpret_tensor(buf32, (1, 512, 768), (393216, 768, 1), 0); del buf32  # reuse
        buf40 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf430 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_3, attention_output, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf34, buf28, primals_14, primals_15, primals_20, primals_21, buf39, buf40, buf430, 512, 768, grid=grid(512), stream=stream0)
        del primals_15
        buf41 = reinterpret_tensor(buf34, (512, 768), (768, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf40, reinterpret_tensor(primals_22, (768, 768), (1, 768), 0), out=buf41)
        buf42 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf40, reinterpret_tensor(primals_24, (768, 768), (1, 768), 0), out=buf42)
        buf43 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf40, reinterpret_tensor(primals_26, (768, 768), (1, 768), 0), out=buf43)
        buf44 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf41, primals_23, buf44, 393216, grid=grid(393216), stream=stream0)
        del primals_23
        buf45 = reinterpret_tensor(buf41, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf42, primals_25, buf45, 393216, grid=grid(393216), stream=stream0)
        del primals_25
        buf46 = reinterpret_tensor(buf42, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf43, primals_27, buf46, 393216, grid=grid(393216), stream=stream0)
        del primals_27
        # Source Nodes: [], Original ATen: []
        buf47 = aten._scaled_dot_product_efficient_attention(buf44, buf45, buf46, None, True, 0.1, scale=0.125)
        buf48 = buf47[0]
        buf49 = buf47[1]
        buf50 = buf47[2]
        buf51 = buf47[3]
        del buf47
        buf52 = buf43; del buf43  # reuse
        # Source Nodes: [hidden_states_9], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf48, buf52, 393216, grid=grid(393216), stream=stream0)
        buf53 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, buf52, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
        del primals_29
        # Source Nodes: [hidden_states_10], Original ATen: [aten.native_dropout]
        buf54 = aten.native_dropout(reinterpret_tensor(buf53, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf55 = buf54[0]
        buf56 = buf54[1]
        del buf54
        buf60 = reinterpret_tensor(buf53, (1, 512, 768), (393216, 768, 1), 0); del buf53  # reuse
        buf61 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf429 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, attention_output_2, hidden_states_12, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf55, buf39, primals_20, primals_21, primals_30, primals_31, buf60, buf61, buf429, 512, 768, grid=grid(512), stream=stream0)
        del primals_21
        buf62 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf61, reinterpret_tensor(primals_32, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf62)
        del primals_33
        buf63 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf62, buf63, 1572864, grid=grid(1572864), stream=stream0)
        buf64 = reinterpret_tensor(buf55, (512, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, buf63, reinterpret_tensor(primals_34, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf64)
        del primals_35
        # Source Nodes: [hidden_states_15], Original ATen: [aten.native_dropout]
        buf65 = aten.native_dropout(reinterpret_tensor(buf64, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf66 = buf65[0]
        buf67 = buf65[1]
        del buf65
        buf71 = reinterpret_tensor(buf64, (1, 512, 768), (393216, 768, 1), 0); del buf64  # reuse
        buf72 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf428 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, attention_output_2, hidden_states_17, mixed_query_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf66, buf60, primals_30, primals_31, primals_36, primals_37, buf71, buf72, buf428, 512, 768, grid=grid(512), stream=stream0)
        del primals_31
        buf73 = reinterpret_tensor(buf66, (512, 768), (768, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf72, reinterpret_tensor(primals_38, (768, 768), (1, 768), 0), out=buf73)
        buf74 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf72, reinterpret_tensor(primals_40, (768, 768), (1, 768), 0), out=buf74)
        buf75 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf72, reinterpret_tensor(primals_42, (768, 768), (1, 768), 0), out=buf75)
        buf76 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf73, primals_39, buf76, 393216, grid=grid(393216), stream=stream0)
        del primals_39
        buf77 = reinterpret_tensor(buf73, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf74, primals_41, buf77, 393216, grid=grid(393216), stream=stream0)
        del primals_41
        buf78 = reinterpret_tensor(buf74, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf75, primals_43, buf78, 393216, grid=grid(393216), stream=stream0)
        del primals_43
        # Source Nodes: [], Original ATen: []
        buf79 = aten._scaled_dot_product_efficient_attention(buf76, buf77, buf78, None, True, 0.1, scale=0.125)
        buf80 = buf79[0]
        buf81 = buf79[1]
        buf82 = buf79[2]
        buf83 = buf79[3]
        del buf79
        buf84 = buf75; del buf75  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf80, buf84, 393216, grid=grid(393216), stream=stream0)
        buf85 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, buf84, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf85)
        del primals_45
        # Source Nodes: [hidden_states_19], Original ATen: [aten.native_dropout]
        buf86 = aten.native_dropout(reinterpret_tensor(buf85, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf87 = buf86[0]
        buf88 = buf86[1]
        del buf86
        buf92 = reinterpret_tensor(buf85, (1, 512, 768), (393216, 768, 1), 0); del buf85  # reuse
        buf93 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf427 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, attention_output_4, hidden_states_17, hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf87, buf71, primals_36, primals_37, primals_46, primals_47, buf92, buf93, buf427, 512, 768, grid=grid(512), stream=stream0)
        del primals_37
        buf94 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, buf93, reinterpret_tensor(primals_48, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf94)
        del primals_49
        buf95 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_23, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf94, buf95, 1572864, grid=grid(1572864), stream=stream0)
        buf96 = reinterpret_tensor(buf87, (512, 768), (768, 1), 0); del buf87  # reuse
        # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_51, buf95, reinterpret_tensor(primals_50, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf96)
        del primals_51
        # Source Nodes: [hidden_states_24], Original ATen: [aten.native_dropout]
        buf97 = aten.native_dropout(reinterpret_tensor(buf96, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf98 = buf97[0]
        buf99 = buf97[1]
        del buf97
        buf103 = reinterpret_tensor(buf96, (1, 512, 768), (393216, 768, 1), 0); del buf96  # reuse
        buf104 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf426 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, attention_output_4, hidden_states_26, mixed_query_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf98, buf92, primals_46, primals_47, primals_52, primals_53, buf103, buf104, buf426, 512, 768, grid=grid(512), stream=stream0)
        del primals_47
        buf105 = reinterpret_tensor(buf98, (512, 768), (768, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf104, reinterpret_tensor(primals_54, (768, 768), (1, 768), 0), out=buf105)
        buf106 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf104, reinterpret_tensor(primals_56, (768, 768), (1, 768), 0), out=buf106)
        buf107 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf104, reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), out=buf107)
        buf108 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf105, primals_55, buf108, 393216, grid=grid(393216), stream=stream0)
        del primals_55
        buf109 = reinterpret_tensor(buf105, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf106, primals_57, buf109, 393216, grid=grid(393216), stream=stream0)
        del primals_57
        buf110 = reinterpret_tensor(buf106, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf107, primals_59, buf110, 393216, grid=grid(393216), stream=stream0)
        del primals_59
        # Source Nodes: [], Original ATen: []
        buf111 = aten._scaled_dot_product_efficient_attention(buf108, buf109, buf110, None, True, 0.1, scale=0.125)
        buf112 = buf111[0]
        buf113 = buf111[1]
        buf114 = buf111[2]
        buf115 = buf111[3]
        del buf111
        buf116 = buf107; del buf107  # reuse
        # Source Nodes: [hidden_states_27], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf112, buf116, 393216, grid=grid(393216), stream=stream0)
        buf117 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_61, buf116, reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf117)
        del primals_61
        # Source Nodes: [hidden_states_28], Original ATen: [aten.native_dropout]
        buf118 = aten.native_dropout(reinterpret_tensor(buf117, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf119 = buf118[0]
        buf120 = buf118[1]
        del buf118
        buf124 = reinterpret_tensor(buf117, (1, 512, 768), (393216, 768, 1), 0); del buf117  # reuse
        buf125 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf425 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, attention_output_6, hidden_states_26, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf119, buf103, primals_52, primals_53, primals_62, primals_63, buf124, buf125, buf425, 512, 768, grid=grid(512), stream=stream0)
        del primals_53
        buf126 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf125, reinterpret_tensor(primals_64, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf126)
        del primals_65
        buf127 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf126, buf127, 1572864, grid=grid(1572864), stream=stream0)
        buf128 = reinterpret_tensor(buf119, (512, 768), (768, 1), 0); del buf119  # reuse
        # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_67, buf127, reinterpret_tensor(primals_66, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf128)
        del primals_67
        # Source Nodes: [hidden_states_33], Original ATen: [aten.native_dropout]
        buf129 = aten.native_dropout(reinterpret_tensor(buf128, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf130 = buf129[0]
        buf131 = buf129[1]
        del buf129
        buf135 = reinterpret_tensor(buf128, (1, 512, 768), (393216, 768, 1), 0); del buf128  # reuse
        buf136 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf424 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, attention_output_6, hidden_states_35, mixed_query_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf130, buf124, primals_62, primals_63, primals_68, primals_69, buf135, buf136, buf424, 512, 768, grid=grid(512), stream=stream0)
        del primals_63
        buf137 = reinterpret_tensor(buf130, (512, 768), (768, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf136, reinterpret_tensor(primals_70, (768, 768), (1, 768), 0), out=buf137)
        buf138 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf136, reinterpret_tensor(primals_72, (768, 768), (1, 768), 0), out=buf138)
        buf139 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf136, reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), out=buf139)
        buf140 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf137, primals_71, buf140, 393216, grid=grid(393216), stream=stream0)
        del primals_71
        buf141 = reinterpret_tensor(buf137, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf138, primals_73, buf141, 393216, grid=grid(393216), stream=stream0)
        del primals_73
        buf142 = reinterpret_tensor(buf138, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf139, primals_75, buf142, 393216, grid=grid(393216), stream=stream0)
        del primals_75
        # Source Nodes: [], Original ATen: []
        buf143 = aten._scaled_dot_product_efficient_attention(buf140, buf141, buf142, None, True, 0.1, scale=0.125)
        buf144 = buf143[0]
        buf145 = buf143[1]
        buf146 = buf143[2]
        buf147 = buf143[3]
        del buf143
        buf148 = buf139; del buf139  # reuse
        # Source Nodes: [hidden_states_36], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf144, buf148, 393216, grid=grid(393216), stream=stream0)
        buf149 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, buf148, reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf149)
        del primals_77
        # Source Nodes: [hidden_states_37], Original ATen: [aten.native_dropout]
        buf150 = aten.native_dropout(reinterpret_tensor(buf149, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf151 = buf150[0]
        buf152 = buf150[1]
        del buf150
        buf156 = reinterpret_tensor(buf149, (1, 512, 768), (393216, 768, 1), 0); del buf149  # reuse
        buf157 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf423 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, attention_output_8, hidden_states_35, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf151, buf135, primals_68, primals_69, primals_78, primals_79, buf156, buf157, buf423, 512, 768, grid=grid(512), stream=stream0)
        del primals_69
        buf158 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf157, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf158)
        del primals_81
        buf159 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_41, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf158, buf159, 1572864, grid=grid(1572864), stream=stream0)
        buf160 = reinterpret_tensor(buf151, (512, 768), (768, 1), 0); del buf151  # reuse
        # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_83, buf159, reinterpret_tensor(primals_82, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf160)
        del primals_83
        # Source Nodes: [hidden_states_42], Original ATen: [aten.native_dropout]
        buf161 = aten.native_dropout(reinterpret_tensor(buf160, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf162 = buf161[0]
        buf163 = buf161[1]
        del buf161
        buf167 = reinterpret_tensor(buf160, (1, 512, 768), (393216, 768, 1), 0); del buf160  # reuse
        buf168 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf422 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, attention_output_8, hidden_states_44, mixed_query_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf162, buf156, primals_78, primals_79, primals_84, primals_85, buf167, buf168, buf422, 512, 768, grid=grid(512), stream=stream0)
        del primals_79
        buf169 = reinterpret_tensor(buf162, (512, 768), (768, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf168, reinterpret_tensor(primals_86, (768, 768), (1, 768), 0), out=buf169)
        buf170 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf168, reinterpret_tensor(primals_88, (768, 768), (1, 768), 0), out=buf170)
        buf171 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf168, reinterpret_tensor(primals_90, (768, 768), (1, 768), 0), out=buf171)
        buf172 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf169, primals_87, buf172, 393216, grid=grid(393216), stream=stream0)
        del primals_87
        buf173 = reinterpret_tensor(buf169, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf170, primals_89, buf173, 393216, grid=grid(393216), stream=stream0)
        del primals_89
        buf174 = reinterpret_tensor(buf170, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf171, primals_91, buf174, 393216, grid=grid(393216), stream=stream0)
        del primals_91
        # Source Nodes: [], Original ATen: []
        buf175 = aten._scaled_dot_product_efficient_attention(buf172, buf173, buf174, None, True, 0.1, scale=0.125)
        buf176 = buf175[0]
        buf177 = buf175[1]
        buf178 = buf175[2]
        buf179 = buf175[3]
        del buf175
        buf180 = buf171; del buf171  # reuse
        # Source Nodes: [hidden_states_45], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf176, buf180, 393216, grid=grid(393216), stream=stream0)
        buf181 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, buf180, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf181)
        del primals_93
        # Source Nodes: [hidden_states_46], Original ATen: [aten.native_dropout]
        buf182 = aten.native_dropout(reinterpret_tensor(buf181, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        buf188 = reinterpret_tensor(buf181, (1, 512, 768), (393216, 768, 1), 0); del buf181  # reuse
        buf189 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf421 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, attention_output_10, hidden_states_44, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf183, buf167, primals_84, primals_85, primals_94, primals_95, buf188, buf189, buf421, 512, 768, grid=grid(512), stream=stream0)
        del primals_85
        buf190 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, buf189, reinterpret_tensor(primals_96, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf190)
        del primals_97
        buf191 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf190, buf191, 1572864, grid=grid(1572864), stream=stream0)
        buf192 = reinterpret_tensor(buf183, (512, 768), (768, 1), 0); del buf183  # reuse
        # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_99, buf191, reinterpret_tensor(primals_98, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf192)
        del primals_99
        # Source Nodes: [hidden_states_51], Original ATen: [aten.native_dropout]
        buf193 = aten.native_dropout(reinterpret_tensor(buf192, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf194 = buf193[0]
        buf195 = buf193[1]
        del buf193
        buf199 = reinterpret_tensor(buf192, (1, 512, 768), (393216, 768, 1), 0); del buf192  # reuse
        buf200 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf420 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, attention_output_10, hidden_states_53, mixed_query_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf194, buf188, primals_94, primals_95, primals_100, primals_101, buf199, buf200, buf420, 512, 768, grid=grid(512), stream=stream0)
        del primals_95
        buf201 = reinterpret_tensor(buf194, (512, 768), (768, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf200, reinterpret_tensor(primals_102, (768, 768), (1, 768), 0), out=buf201)
        buf202 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf200, reinterpret_tensor(primals_104, (768, 768), (1, 768), 0), out=buf202)
        buf203 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf200, reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), out=buf203)
        buf204 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf201, primals_103, buf204, 393216, grid=grid(393216), stream=stream0)
        del primals_103
        buf205 = reinterpret_tensor(buf201, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf202, primals_105, buf205, 393216, grid=grid(393216), stream=stream0)
        del primals_105
        buf206 = reinterpret_tensor(buf202, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf203, primals_107, buf206, 393216, grid=grid(393216), stream=stream0)
        del primals_107
        # Source Nodes: [], Original ATen: []
        buf207 = aten._scaled_dot_product_efficient_attention(buf204, buf205, buf206, None, True, 0.1, scale=0.125)
        buf208 = buf207[0]
        buf209 = buf207[1]
        buf210 = buf207[2]
        buf211 = buf207[3]
        del buf207
        buf212 = buf203; del buf203  # reuse
        # Source Nodes: [hidden_states_54], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf208, buf212, 393216, grid=grid(393216), stream=stream0)
        buf213 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_109, buf212, reinterpret_tensor(primals_108, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf213)
        del primals_109
        # Source Nodes: [hidden_states_55], Original ATen: [aten.native_dropout]
        buf214 = aten.native_dropout(reinterpret_tensor(buf213, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        buf220 = reinterpret_tensor(buf213, (1, 512, 768), (393216, 768, 1), 0); del buf213  # reuse
        buf221 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf419 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, attention_output_12, hidden_states_53, hidden_states_57], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf215, buf199, primals_100, primals_101, primals_110, primals_111, buf220, buf221, buf419, 512, 768, grid=grid(512), stream=stream0)
        del primals_101
        buf222 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_113, buf221, reinterpret_tensor(primals_112, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf222)
        del primals_113
        buf223 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf222, buf223, 1572864, grid=grid(1572864), stream=stream0)
        buf224 = reinterpret_tensor(buf215, (512, 768), (768, 1), 0); del buf215  # reuse
        # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_115, buf223, reinterpret_tensor(primals_114, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf224)
        del primals_115
        # Source Nodes: [hidden_states_60], Original ATen: [aten.native_dropout]
        buf225 = aten.native_dropout(reinterpret_tensor(buf224, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf226 = buf225[0]
        buf227 = buf225[1]
        del buf225
        buf231 = reinterpret_tensor(buf224, (1, 512, 768), (393216, 768, 1), 0); del buf224  # reuse
        buf232 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf418 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, attention_output_12, hidden_states_62, mixed_query_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf226, buf220, primals_110, primals_111, primals_116, primals_117, buf231, buf232, buf418, 512, 768, grid=grid(512), stream=stream0)
        del primals_111
        buf233 = reinterpret_tensor(buf226, (512, 768), (768, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf232, reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), out=buf233)
        buf234 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf232, reinterpret_tensor(primals_120, (768, 768), (1, 768), 0), out=buf234)
        buf235 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf232, reinterpret_tensor(primals_122, (768, 768), (1, 768), 0), out=buf235)
        buf236 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf233, primals_119, buf236, 393216, grid=grid(393216), stream=stream0)
        del primals_119
        buf237 = reinterpret_tensor(buf233, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf234, primals_121, buf237, 393216, grid=grid(393216), stream=stream0)
        del primals_121
        buf238 = reinterpret_tensor(buf234, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf235, primals_123, buf238, 393216, grid=grid(393216), stream=stream0)
        del primals_123
        # Source Nodes: [], Original ATen: []
        buf239 = aten._scaled_dot_product_efficient_attention(buf236, buf237, buf238, None, True, 0.1, scale=0.125)
        buf240 = buf239[0]
        buf241 = buf239[1]
        buf242 = buf239[2]
        buf243 = buf239[3]
        del buf239
        buf244 = buf235; del buf235  # reuse
        # Source Nodes: [hidden_states_63], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf240, buf244, 393216, grid=grid(393216), stream=stream0)
        buf245 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_125, buf244, reinterpret_tensor(primals_124, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf245)
        del primals_125
        # Source Nodes: [hidden_states_64], Original ATen: [aten.native_dropout]
        buf246 = aten.native_dropout(reinterpret_tensor(buf245, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf247 = buf246[0]
        buf248 = buf246[1]
        del buf246
        buf252 = reinterpret_tensor(buf245, (1, 512, 768), (393216, 768, 1), 0); del buf245  # reuse
        buf253 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf417 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, attention_output_14, hidden_states_62, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf247, buf231, primals_116, primals_117, primals_126, primals_127, buf252, buf253, buf417, 512, 768, grid=grid(512), stream=stream0)
        del primals_117
        buf254 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf253, reinterpret_tensor(primals_128, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf254)
        del primals_129
        buf255 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_68, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf254, buf255, 1572864, grid=grid(1572864), stream=stream0)
        buf256 = reinterpret_tensor(buf247, (512, 768), (768, 1), 0); del buf247  # reuse
        # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_131, buf255, reinterpret_tensor(primals_130, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf256)
        del primals_131
        # Source Nodes: [hidden_states_69], Original ATen: [aten.native_dropout]
        buf257 = aten.native_dropout(reinterpret_tensor(buf256, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf258 = buf257[0]
        buf259 = buf257[1]
        del buf257
        buf263 = reinterpret_tensor(buf256, (1, 512, 768), (393216, 768, 1), 0); del buf256  # reuse
        buf264 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf416 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, attention_output_14, hidden_states_71, mixed_query_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf258, buf252, primals_126, primals_127, primals_132, primals_133, buf263, buf264, buf416, 512, 768, grid=grid(512), stream=stream0)
        del primals_127
        buf265 = reinterpret_tensor(buf258, (512, 768), (768, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf264, reinterpret_tensor(primals_134, (768, 768), (1, 768), 0), out=buf265)
        buf266 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf264, reinterpret_tensor(primals_136, (768, 768), (1, 768), 0), out=buf266)
        buf267 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf264, reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), out=buf267)
        buf268 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf265, primals_135, buf268, 393216, grid=grid(393216), stream=stream0)
        del primals_135
        buf269 = reinterpret_tensor(buf265, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf266, primals_137, buf269, 393216, grid=grid(393216), stream=stream0)
        del primals_137
        buf270 = reinterpret_tensor(buf266, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf267, primals_139, buf270, 393216, grid=grid(393216), stream=stream0)
        del primals_139
        # Source Nodes: [], Original ATen: []
        buf271 = aten._scaled_dot_product_efficient_attention(buf268, buf269, buf270, None, True, 0.1, scale=0.125)
        buf272 = buf271[0]
        buf273 = buf271[1]
        buf274 = buf271[2]
        buf275 = buf271[3]
        del buf271
        buf276 = buf267; del buf267  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf272, buf276, 393216, grid=grid(393216), stream=stream0)
        buf277 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_141, buf276, reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf277)
        del primals_141
        # Source Nodes: [hidden_states_73], Original ATen: [aten.native_dropout]
        buf278 = aten.native_dropout(reinterpret_tensor(buf277, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf279 = buf278[0]
        buf280 = buf278[1]
        del buf278
        buf284 = reinterpret_tensor(buf277, (1, 512, 768), (393216, 768, 1), 0); del buf277  # reuse
        buf285 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf415 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, attention_output_16, hidden_states_71, hidden_states_75], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf279, buf263, primals_132, primals_133, primals_142, primals_143, buf284, buf285, buf415, 512, 768, grid=grid(512), stream=stream0)
        del primals_133
        buf286 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, buf285, reinterpret_tensor(primals_144, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf286)
        del primals_145
        buf287 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_77, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf286, buf287, 1572864, grid=grid(1572864), stream=stream0)
        buf288 = reinterpret_tensor(buf279, (512, 768), (768, 1), 0); del buf279  # reuse
        # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_147, buf287, reinterpret_tensor(primals_146, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf288)
        del primals_147
        # Source Nodes: [hidden_states_78], Original ATen: [aten.native_dropout]
        buf289 = aten.native_dropout(reinterpret_tensor(buf288, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf290 = buf289[0]
        buf291 = buf289[1]
        del buf289
        buf295 = reinterpret_tensor(buf288, (1, 512, 768), (393216, 768, 1), 0); del buf288  # reuse
        buf296 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf414 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, attention_output_16, hidden_states_80, mixed_query_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf290, buf284, primals_142, primals_143, primals_148, primals_149, buf295, buf296, buf414, 512, 768, grid=grid(512), stream=stream0)
        del primals_143
        buf297 = reinterpret_tensor(buf290, (512, 768), (768, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf296, reinterpret_tensor(primals_150, (768, 768), (1, 768), 0), out=buf297)
        buf298 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf296, reinterpret_tensor(primals_152, (768, 768), (1, 768), 0), out=buf298)
        buf299 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf296, reinterpret_tensor(primals_154, (768, 768), (1, 768), 0), out=buf299)
        buf300 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf297, primals_151, buf300, 393216, grid=grid(393216), stream=stream0)
        del primals_151
        buf301 = reinterpret_tensor(buf297, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf298, primals_153, buf301, 393216, grid=grid(393216), stream=stream0)
        del primals_153
        buf302 = reinterpret_tensor(buf298, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf299, primals_155, buf302, 393216, grid=grid(393216), stream=stream0)
        del primals_155
        # Source Nodes: [], Original ATen: []
        buf303 = aten._scaled_dot_product_efficient_attention(buf300, buf301, buf302, None, True, 0.1, scale=0.125)
        buf304 = buf303[0]
        buf305 = buf303[1]
        buf306 = buf303[2]
        buf307 = buf303[3]
        del buf303
        buf308 = buf299; del buf299  # reuse
        # Source Nodes: [hidden_states_81], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf304, buf308, 393216, grid=grid(393216), stream=stream0)
        buf309 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_157, buf308, reinterpret_tensor(primals_156, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf309)
        del primals_157
        # Source Nodes: [hidden_states_82], Original ATen: [aten.native_dropout]
        buf310 = aten.native_dropout(reinterpret_tensor(buf309, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf311 = buf310[0]
        buf312 = buf310[1]
        del buf310
        buf316 = reinterpret_tensor(buf309, (1, 512, 768), (393216, 768, 1), 0); del buf309  # reuse
        buf317 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf413 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, attention_output_18, hidden_states_80, hidden_states_84], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf311, buf295, primals_148, primals_149, primals_158, primals_159, buf316, buf317, buf413, 512, 768, grid=grid(512), stream=stream0)
        del primals_149
        buf318 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf317, reinterpret_tensor(primals_160, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf318)
        del primals_161
        buf319 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_86, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf318, buf319, 1572864, grid=grid(1572864), stream=stream0)
        buf320 = reinterpret_tensor(buf311, (512, 768), (768, 1), 0); del buf311  # reuse
        # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_163, buf319, reinterpret_tensor(primals_162, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf320)
        del primals_163
        # Source Nodes: [hidden_states_87], Original ATen: [aten.native_dropout]
        buf321 = aten.native_dropout(reinterpret_tensor(buf320, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf322 = buf321[0]
        buf323 = buf321[1]
        del buf321
        buf327 = reinterpret_tensor(buf320, (1, 512, 768), (393216, 768, 1), 0); del buf320  # reuse
        buf328 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf412 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, attention_output_18, hidden_states_89, mixed_query_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf322, buf316, primals_158, primals_159, primals_164, primals_165, buf327, buf328, buf412, 512, 768, grid=grid(512), stream=stream0)
        del primals_159
        buf329 = reinterpret_tensor(buf322, (512, 768), (768, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf328, reinterpret_tensor(primals_166, (768, 768), (1, 768), 0), out=buf329)
        buf330 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf328, reinterpret_tensor(primals_168, (768, 768), (1, 768), 0), out=buf330)
        buf331 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf328, reinterpret_tensor(primals_170, (768, 768), (1, 768), 0), out=buf331)
        buf332 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf329, primals_167, buf332, 393216, grid=grid(393216), stream=stream0)
        del primals_167
        buf333 = reinterpret_tensor(buf329, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf330, primals_169, buf333, 393216, grid=grid(393216), stream=stream0)
        del primals_169
        buf334 = reinterpret_tensor(buf330, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf331, primals_171, buf334, 393216, grid=grid(393216), stream=stream0)
        del primals_171
        # Source Nodes: [], Original ATen: []
        buf335 = aten._scaled_dot_product_efficient_attention(buf332, buf333, buf334, None, True, 0.1, scale=0.125)
        buf336 = buf335[0]
        buf337 = buf335[1]
        buf338 = buf335[2]
        buf339 = buf335[3]
        del buf335
        buf340 = buf331; del buf331  # reuse
        # Source Nodes: [hidden_states_90], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf336, buf340, 393216, grid=grid(393216), stream=stream0)
        buf341 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_173, buf340, reinterpret_tensor(primals_172, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf341)
        del primals_173
        # Source Nodes: [hidden_states_91], Original ATen: [aten.native_dropout]
        buf342 = aten.native_dropout(reinterpret_tensor(buf341, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf343 = buf342[0]
        buf344 = buf342[1]
        del buf342
        buf348 = reinterpret_tensor(buf341, (1, 512, 768), (393216, 768, 1), 0); del buf341  # reuse
        buf349 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf411 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32, attention_output_20, hidden_states_89, hidden_states_93], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf343, buf327, primals_164, primals_165, primals_174, primals_175, buf348, buf349, buf411, 512, 768, grid=grid(512), stream=stream0)
        del primals_165
        buf350 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf349, reinterpret_tensor(primals_176, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf350)
        del primals_177
        buf351 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf350, buf351, 1572864, grid=grid(1572864), stream=stream0)
        buf352 = reinterpret_tensor(buf343, (512, 768), (768, 1), 0); del buf343  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_179, buf351, reinterpret_tensor(primals_178, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf352)
        del primals_179
        # Source Nodes: [hidden_states_96], Original ATen: [aten.native_dropout]
        buf353 = aten.native_dropout(reinterpret_tensor(buf352, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf354 = buf353[0]
        buf355 = buf353[1]
        del buf353
        buf359 = reinterpret_tensor(buf352, (1, 512, 768), (393216, 768, 1), 0); del buf352  # reuse
        buf360 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf410 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, attention_output_20, hidden_states_98, mixed_query_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf354, buf348, primals_174, primals_175, primals_180, primals_181, buf359, buf360, buf410, 512, 768, grid=grid(512), stream=stream0)
        del primals_175
        buf361 = reinterpret_tensor(buf354, (512, 768), (768, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf360, reinterpret_tensor(primals_182, (768, 768), (1, 768), 0), out=buf361)
        buf362 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf360, reinterpret_tensor(primals_184, (768, 768), (1, 768), 0), out=buf362)
        buf363 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf360, reinterpret_tensor(primals_186, (768, 768), (1, 768), 0), out=buf363)
        buf364 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf361, primals_183, buf364, 393216, grid=grid(393216), stream=stream0)
        del primals_183
        buf365 = reinterpret_tensor(buf361, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf362, primals_185, buf365, 393216, grid=grid(393216), stream=stream0)
        del primals_185
        buf366 = reinterpret_tensor(buf362, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(buf363, primals_187, buf366, 393216, grid=grid(393216), stream=stream0)
        del primals_187
        # Source Nodes: [], Original ATen: []
        buf367 = aten._scaled_dot_product_efficient_attention(buf364, buf365, buf366, None, True, 0.1, scale=0.125)
        buf368 = buf367[0]
        buf369 = buf367[1]
        buf370 = buf367[2]
        buf371 = buf367[3]
        del buf367
        buf372 = buf363; del buf363  # reuse
        # Source Nodes: [hidden_states_99], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf368, buf372, 393216, grid=grid(393216), stream=stream0)
        buf373 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_189, buf372, reinterpret_tensor(primals_188, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf373)
        del primals_189
        # Source Nodes: [hidden_states_100], Original ATen: [aten.native_dropout]
        buf374 = aten.native_dropout(reinterpret_tensor(buf373, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf375 = buf374[0]
        buf376 = buf374[1]
        del buf374
        buf380 = reinterpret_tensor(buf373, (1, 512, 768), (393216, 768, 1), 0); del buf373  # reuse
        buf381 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf409 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35, attention_output_22, hidden_states_102, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf375, buf359, primals_180, primals_181, primals_190, primals_191, buf380, buf381, buf409, 512, 768, grid=grid(512), stream=stream0)
        del primals_181
        buf382 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_193, buf381, reinterpret_tensor(primals_192, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf382)
        del primals_193
        buf383 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_104, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf382, buf383, 1572864, grid=grid(1572864), stream=stream0)
        buf384 = reinterpret_tensor(buf375, (512, 768), (768, 1), 0); del buf375  # reuse
        # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_195, buf383, reinterpret_tensor(primals_194, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf384)
        del primals_195
        # Source Nodes: [hidden_states_105], Original ATen: [aten.native_dropout]
        buf385 = aten.native_dropout(reinterpret_tensor(buf384, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf386 = buf385[0]
        buf387 = buf385[1]
        del buf385
        buf391 = reinterpret_tensor(buf384, (1, 512, 768), (393216, 768, 1), 0); del buf384  # reuse
        buf392 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf408 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, attention_output_22, logits, sequence_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf386, buf380, primals_190, primals_191, primals_196, primals_197, buf391, buf392, buf408, 512, 768, grid=grid(512), stream=stream0)
        del buf386
        del primals_191
        del primals_197
        buf393 = empty((512, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf392, reinterpret_tensor(primals_198, (768, 2), (1, 768), 0), out=buf393)
        buf394 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf398 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_logits_1, start_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_6.run(buf393, primals_199, buf394, buf398, 1, 512, grid=grid(1), stream=stream0)
        buf395 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf402 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_logits_1, end_loss], Original ATen: [aten._log_softmax, aten.clone]
        triton_per_fused__log_softmax_clone_7.run(buf393, primals_199, buf395, buf402, 1, 512, grid=grid(1), stream=stream0)
        del buf393
        del primals_199
        buf399 = empty((1, ), device='cuda', dtype=torch.bool)
        buf403 = empty((1, ), device='cuda', dtype=torch.bool)
        buf433 = empty((), device='cuda', dtype=torch.float32)
        buf404 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf405 = empty((1, 1), device='cuda', dtype=torch.int64)
        buf406 = empty((1, 1), device='cuda', dtype=torch.bool)
        buf407 = empty((1, 1), device='cuda', dtype=torch.int64)
        # Source Nodes: [add_37, end_loss, end_positions, loss, start_loss, start_positions], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_add_clamp_div_nll_loss_backward_nll_loss_forward_8.run(primals_203, primals_204, buf398, buf402, buf399, buf403, buf433, buf404, buf405, buf406, buf407, 1, grid=grid(1), stream=stream0)
        del primals_203
        del primals_204
        return (buf433, buf394, buf395, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_202, primals_200, primals_201, buf4, buf8, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), buf12, buf13, buf14, buf17, buf18, buf19, buf16, buf20, buf24, buf28, buf29, buf30, buf31, buf35, buf39, buf40, buf44, buf45, buf46, buf49, buf50, buf51, buf48, buf52, buf56, buf60, buf61, buf62, buf63, buf67, buf71, buf72, buf76, buf77, buf78, buf81, buf82, buf83, buf80, buf84, buf88, buf92, buf93, buf94, buf95, buf99, buf103, buf104, buf108, buf109, buf110, buf113, buf114, buf115, buf112, buf116, buf120, buf124, buf125, buf126, buf127, buf131, buf135, buf136, buf140, buf141, buf142, buf145, buf146, buf147, buf144, buf148, buf152, buf156, buf157, buf158, buf159, buf163, buf167, buf168, buf172, buf173, buf174, buf177, buf178, buf179, buf176, buf180, buf184, buf188, buf189, buf190, buf191, buf195, buf199, buf200, buf204, buf205, buf206, buf209, buf210, buf211, buf208, buf212, buf216, buf220, buf221, buf222, buf223, buf227, buf231, buf232, buf236, buf237, buf238, buf241, buf242, buf243, buf240, buf244, buf248, buf252, buf253, buf254, buf255, buf259, buf263, buf264, buf268, buf269, buf270, buf273, buf274, buf275, buf272, buf276, buf280, buf284, buf285, buf286, buf287, buf291, buf295, buf296, buf300, buf301, buf302, buf305, buf306, buf307, buf304, buf308, buf312, buf316, buf317, buf318, buf319, buf323, buf327, buf328, buf332, buf333, buf334, buf337, buf338, buf339, buf336, buf340, buf344, buf348, buf349, buf350, buf351, buf355, buf359, buf360, buf364, buf365, buf366, buf369, buf370, buf371, buf368, buf372, buf376, buf380, buf381, buf382, buf383, buf387, buf391, buf392, buf398, buf399, buf402, buf403, buf404, buf405, buf406, buf407, reinterpret_tensor(primals_198, (2, 768), (768, 1), 0), buf408, reinterpret_tensor(primals_194, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_192, (3072, 768), (768, 1), 0), buf409, reinterpret_tensor(primals_188, (768, 768), (768, 1), 0), reinterpret_tensor(primals_186, (768, 768), (768, 1), 0), reinterpret_tensor(primals_184, (768, 768), (768, 1), 0), reinterpret_tensor(primals_182, (768, 768), (768, 1), 0), buf410, reinterpret_tensor(primals_178, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_176, (3072, 768), (768, 1), 0), buf411, reinterpret_tensor(primals_172, (768, 768), (768, 1), 0), reinterpret_tensor(primals_170, (768, 768), (768, 1), 0), reinterpret_tensor(primals_168, (768, 768), (768, 1), 0), reinterpret_tensor(primals_166, (768, 768), (768, 1), 0), buf412, reinterpret_tensor(primals_162, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_160, (3072, 768), (768, 1), 0), buf413, reinterpret_tensor(primals_156, (768, 768), (768, 1), 0), reinterpret_tensor(primals_154, (768, 768), (768, 1), 0), reinterpret_tensor(primals_152, (768, 768), (768, 1), 0), reinterpret_tensor(primals_150, (768, 768), (768, 1), 0), buf414, reinterpret_tensor(primals_146, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_144, (3072, 768), (768, 1), 0), buf415, reinterpret_tensor(primals_140, (768, 768), (768, 1), 0), reinterpret_tensor(primals_138, (768, 768), (768, 1), 0), reinterpret_tensor(primals_136, (768, 768), (768, 1), 0), reinterpret_tensor(primals_134, (768, 768), (768, 1), 0), buf416, reinterpret_tensor(primals_130, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_128, (3072, 768), (768, 1), 0), buf417, reinterpret_tensor(primals_124, (768, 768), (768, 1), 0), reinterpret_tensor(primals_122, (768, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 768), (768, 1), 0), reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), buf418, reinterpret_tensor(primals_114, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_112, (3072, 768), (768, 1), 0), buf419, reinterpret_tensor(primals_108, (768, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), reinterpret_tensor(primals_104, (768, 768), (768, 1), 0), reinterpret_tensor(primals_102, (768, 768), (768, 1), 0), buf420, reinterpret_tensor(primals_98, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_96, (3072, 768), (768, 1), 0), buf421, reinterpret_tensor(primals_92, (768, 768), (768, 1), 0), reinterpret_tensor(primals_90, (768, 768), (768, 1), 0), reinterpret_tensor(primals_88, (768, 768), (768, 1), 0), reinterpret_tensor(primals_86, (768, 768), (768, 1), 0), buf422, reinterpret_tensor(primals_82, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 768), (768, 1), 0), buf423, reinterpret_tensor(primals_76, (768, 768), (768, 1), 0), reinterpret_tensor(primals_74, (768, 768), (768, 1), 0), reinterpret_tensor(primals_72, (768, 768), (768, 1), 0), reinterpret_tensor(primals_70, (768, 768), (768, 1), 0), buf424, reinterpret_tensor(primals_66, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_64, (3072, 768), (768, 1), 0), buf425, reinterpret_tensor(primals_60, (768, 768), (768, 1), 0), reinterpret_tensor(primals_58, (768, 768), (768, 1), 0), reinterpret_tensor(primals_56, (768, 768), (768, 1), 0), reinterpret_tensor(primals_54, (768, 768), (768, 1), 0), buf426, reinterpret_tensor(primals_50, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_48, (3072, 768), (768, 1), 0), buf427, reinterpret_tensor(primals_44, (768, 768), (768, 1), 0), reinterpret_tensor(primals_42, (768, 768), (768, 1), 0), reinterpret_tensor(primals_40, (768, 768), (768, 1), 0), reinterpret_tensor(primals_38, (768, 768), (768, 1), 0), buf428, reinterpret_tensor(primals_34, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_32, (3072, 768), (768, 1), 0), buf429, reinterpret_tensor(primals_28, (768, 768), (768, 1), 0), reinterpret_tensor(primals_26, (768, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 768), (768, 1), 0), buf430, reinterpret_tensor(primals_18, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_16, (3072, 768), (768, 1), 0), buf431, reinterpret_tensor(primals_12, (768, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (768, 1), 0), reinterpret_tensor(primals_6, (768, 768), (768, 1), 0), buf432, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_203 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BertForQuestionAnswering', benchmark_compiled_module)
