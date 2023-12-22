
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


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gmguyjzl35y52vobhacdogcxj4bzu2rol3fqph5morsk2pc3v4.py
# Source Nodes: [token_type_ids], Original ATen: [aten.zeros]
# token_type_ids => full_default
triton_poi_fused_zeros_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/zh/czhoc24qwp7ogttflxim4ke2tobeotb2o6ew5atbrmb2klvi4svm.py
# Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, embeddings, embeddings_1, h_position_embeddings, left_position_embeddings, lower_position_embeddings, position_embeddings, right_position_embeddings, token_type_embeddings, upper_position_embeddings, w_position_embeddings, words_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# add_4 => add_4
# add_5 => add_5
# add_6 => add_6
# embeddings => add_7
# embeddings_1 => add_8, add_9, mul_1, mul_2, rsqrt, sub_3, var_mean
# h_position_embeddings => embedding_6
# left_position_embeddings => embedding_2
# lower_position_embeddings => embedding_5
# position_embeddings => embedding_1
# right_position_embeddings => embedding_4
# token_type_embeddings => embedding_8
# upper_position_embeddings => embedding_3
# w_position_embeddings => embedding_7
# words_embeddings => embedding
triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    tmp11 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr8 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr9 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr10 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 30522
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 30522)) | ~xmask, "index out of bounds: 0 <= tmp3 < 30522")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp5 + 512
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert(((0 <= tmp8) & (tmp8 < 512)) | ~xmask, "index out of bounds: 0 <= tmp8 < 512")
    tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, other=0.0)
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14 + tmp11
    tmp16 = tmp15 + tmp13
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
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
    tmp42 = 1e-12
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tmp44 / tmp40
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp18, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp45, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp49, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp50, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztmswonzytrs4gr6tiyjolh437pxweu4zijl5gf5flkr5jtxn3g.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/as/case4br6ehesd7n6gwk5dh5psrrhyia5llkf7l3qkv2ejwr5qgrv.py
# Source Nodes: [hidden_states], Original ATen: [aten.view]
# hidden_states => view_16
triton_poi_fused_view_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tu/ctuqjbkdgp37a46pkiefawoo6lqewb2f5ydv7ak5r2vvapdfp6o7.py
# Source Nodes: [add_9, attention_output, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_9 => add_11
# attention_output => add_12, add_13, mul_3, mul_4, rsqrt_1, sub_5, var_mean_1
# hidden_states_3 => view_18
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7vyqxuzxrmtlgg2xidaeludltrdfephodm3jh46ci6fx2psou4.py
# Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
# hidden_states_5 => view_20
# intermediate_output => add_14, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/44/c44aqzp6nja5ez4l6ohtzypll36f2d7y54feyqxdrj43urkvitai.py
# Source Nodes: [add_10, attention_output, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_10 => add_15
# attention_output => add_13, mul_4
# hidden_states_8 => add_16, add_17, mul_8, mul_9, rsqrt_2, sub_6, var_mean_2
# mixed_query_layer_1 => view_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ay/cayd4nhcpolsq5kuoukj3v54otfofsn7xh6verys4gjjqxbszb4j.py
# Source Nodes: [hidden_states_109, hidden_states_111, prediction_scores], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_109 => add_106, erf_12, mul_87, mul_88, mul_89
# hidden_states_111 => add_107, add_108, mul_90, mul_91, rsqrt_25, sub_40, var_mean_25
# prediction_scores => view_266
triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp32 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
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
    tmp28 = 1e-12
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl73pi3o6uhh3ca475yb4jufzqukmh2shnvnizh7x5tbqc3oqfga.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_12, exp_12, log, sub_41, sub_42, sum_13
triton_red_fused__log_softmax_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 30522
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
        tmp0 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (30522*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.log(tmp8)
        tmp13 = tmp11 - tmp12
        tl.store(out_ptr2 + (r1 + (30522*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/cio7d2ndreov3o7sckivbe7k6etvu6ecrzcl63lkykkudzietrc4.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type, div_24, full_default_5, ne, neg, sum_14, sum_15, where_1
triton_per_fused_nll_loss_forward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.where(tmp2, tmp0, tmp8)
    tmp10 = tmp9 + 30522
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 30522), "index out of bounds: 0 <= tmp12 < 30522")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (30522*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp7.to(tl.float32)
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp21, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/cowbluim5njj6waegzvjiovqh3muj2jfrd22nktzf4shxww3nwmk.py
# Source Nodes: [bbox], Original ATen: [aten.zeros]
# bbox => full_2
triton_poi_fused_zeros_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212 = args
    args.clear()
    assert_size_stride(primals_1, (30522, 768), (768, 1))
    assert_size_stride(primals_2, (512, 768), (768, 1))
    assert_size_stride(primals_3, (1024, 768), (768, 1))
    assert_size_stride(primals_4, (1024, 768), (768, 1))
    assert_size_stride(primals_5, (1024, 768), (768, 1))
    assert_size_stride(primals_6, (1024, 768), (768, 1))
    assert_size_stride(primals_7, (2, 768), (768, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, 768), (768, 1))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, 768), (768, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, 768), (768, 1))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (3072, 768), (768, 1))
    assert_size_stride(primals_21, (3072, ), (1, ))
    assert_size_stride(primals_22, (768, 3072), (3072, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, 768), (768, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, 768), (768, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, 768), (768, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, 768), (768, 1))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (3072, 768), (768, 1))
    assert_size_stride(primals_37, (3072, ), (1, ))
    assert_size_stride(primals_38, (768, 3072), (3072, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, 768), (768, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, 768), (768, 1))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, 768), (768, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, 768), (768, 1))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (3072, 768), (768, 1))
    assert_size_stride(primals_53, (3072, ), (1, ))
    assert_size_stride(primals_54, (768, 3072), (3072, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, 768), (768, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, 768), (768, 1))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, 768), (768, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, 768), (768, 1))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (3072, 768), (768, 1))
    assert_size_stride(primals_69, (3072, ), (1, ))
    assert_size_stride(primals_70, (768, 3072), (3072, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, 768), (768, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, 768), (768, 1))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, 768), (768, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, 768), (768, 1))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (3072, 768), (768, 1))
    assert_size_stride(primals_85, (3072, ), (1, ))
    assert_size_stride(primals_86, (768, 3072), (3072, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, 768), (768, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, 768), (768, 1))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, 768), (768, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, 768), (768, 1))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (3072, 768), (768, 1))
    assert_size_stride(primals_101, (3072, ), (1, ))
    assert_size_stride(primals_102, (768, 3072), (3072, 1))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, 768), (768, 1))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, 768), (768, 1))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, 768), (768, 1))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (3072, 768), (768, 1))
    assert_size_stride(primals_117, (3072, ), (1, ))
    assert_size_stride(primals_118, (768, 3072), (3072, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, 768), (768, 1))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, 768), (768, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, 768), (768, 1))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, 768), (768, 1))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (3072, 768), (768, 1))
    assert_size_stride(primals_133, (3072, ), (1, ))
    assert_size_stride(primals_134, (768, 3072), (3072, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, 768), (768, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, 768), (768, 1))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, 768), (768, 1))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, 768), (768, 1))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (3072, 768), (768, 1))
    assert_size_stride(primals_149, (3072, ), (1, ))
    assert_size_stride(primals_150, (768, 3072), (3072, 1))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (768, 768), (768, 1))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (768, 768), (768, 1))
    assert_size_stride(primals_157, (768, ), (1, ))
    assert_size_stride(primals_158, (768, 768), (768, 1))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_160, (768, 768), (768, 1))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (3072, 768), (768, 1))
    assert_size_stride(primals_165, (3072, ), (1, ))
    assert_size_stride(primals_166, (768, 3072), (3072, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (768, 768), (768, 1))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (768, 768), (768, 1))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (768, 768), (768, 1))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (768, 768), (768, 1))
    assert_size_stride(primals_177, (768, ), (1, ))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (3072, 768), (768, 1))
    assert_size_stride(primals_181, (3072, ), (1, ))
    assert_size_stride(primals_182, (768, 3072), (3072, 1))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, 768), (768, 1))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (768, 768), (768, 1))
    assert_size_stride(primals_189, (768, ), (1, ))
    assert_size_stride(primals_190, (768, 768), (768, 1))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, 768), (768, 1))
    assert_size_stride(primals_193, (768, ), (1, ))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (3072, 768), (768, 1))
    assert_size_stride(primals_197, (3072, ), (1, ))
    assert_size_stride(primals_198, (768, 3072), (3072, 1))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (768, 768), (768, 1))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_204, (768, 768), (768, 1))
    assert_size_stride(primals_205, (768, ), (1, ))
    assert_size_stride(primals_206, (768, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (30522, 768), (768, 1))
    assert_size_stride(primals_209, (30522, ), (1, ))
    assert_size_stride(primals_210, (1, 512), (512, 1))
    assert_size_stride(primals_211, (1, 512), (512, 1))
    assert_size_stride(primals_212, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [token_type_ids], Original ATen: [aten.zeros]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_zeros_0.run(buf0, 512, grid=grid(512), stream=stream0)
        buf1 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf6 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf432 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, embeddings, embeddings_1, h_position_embeddings, left_position_embeddings, lower_position_embeddings, position_embeddings, right_position_embeddings, token_type_embeddings, upper_position_embeddings, w_position_embeddings, words_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_1.run(primals_211, primals_1, primals_210, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, buf1, buf5, buf6, buf432, 512, 768, grid=grid(512), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_4
        del primals_5
        del primals_6
        del primals_7
        del primals_9
        # Source Nodes: [embedding_output, embeddings_1], Original ATen: [aten.native_dropout, aten.native_layer_norm]
        buf7 = aten.native_dropout(buf6, 0.1, True)
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        buf10 = reinterpret_tensor(buf6, (512, 768), (768, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf10)
        buf11 = reinterpret_tensor(buf1, (512, 768), (768, 1), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 768), (768, 1), 0), reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf11)
        buf12 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf8, (512, 768), (768, 1), 0), reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf12)
        buf13 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf10, primals_11, buf13, 393216, grid=grid(393216), stream=stream0)
        del primals_11
        buf14 = reinterpret_tensor(buf10, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf11, primals_13, buf14, 393216, grid=grid(393216), stream=stream0)
        del primals_13
        buf15 = reinterpret_tensor(buf11, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf12, primals_15, buf15, 393216, grid=grid(393216), stream=stream0)
        del primals_15
        # Source Nodes: [], Original ATen: []
        buf16 = aten._scaled_dot_product_efficient_attention(buf13, buf14, buf15, None, True, 0.1, scale=0.125)
        buf17 = buf16[0]
        buf18 = buf16[1]
        buf19 = buf16[2]
        buf20 = buf16[3]
        del buf16
        buf21 = buf12; del buf12  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf17, buf21, 393216, grid=grid(393216), stream=stream0)
        buf22 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, buf21, reinterpret_tensor(primals_16, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf22)
        del primals_17
        # Source Nodes: [hidden_states_1], Original ATen: [aten.native_dropout]
        buf23 = aten.native_dropout(reinterpret_tensor(buf22, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf24 = buf23[0]
        buf25 = buf23[1]
        del buf23
        buf29 = reinterpret_tensor(buf22, (1, 512, 768), (393216, 768, 1), 0); del buf22  # reuse
        buf30 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf431 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, attention_output, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4.run(buf24, buf8, primals_18, primals_19, buf29, buf30, buf431, 512, 768, grid=grid(512), stream=stream0)
        buf31 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, buf30, reinterpret_tensor(primals_20, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf31)
        del primals_21
        buf32 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf31, buf32, 1572864, grid=grid(1572864), stream=stream0)
        buf33 = reinterpret_tensor(buf24, (512, 768), (768, 1), 0); del buf24  # reuse
        # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_23, buf32, reinterpret_tensor(primals_22, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf33)
        del primals_23
        # Source Nodes: [hidden_states_6], Original ATen: [aten.native_dropout]
        buf34 = aten.native_dropout(reinterpret_tensor(buf33, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf35 = buf34[0]
        buf36 = buf34[1]
        del buf34
        buf40 = reinterpret_tensor(buf33, (1, 512, 768), (393216, 768, 1), 0); del buf33  # reuse
        buf41 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf430 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, attention_output, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf35, buf29, primals_18, primals_19, primals_24, primals_25, buf40, buf41, buf430, 512, 768, grid=grid(512), stream=stream0)
        del primals_19
        buf42 = reinterpret_tensor(buf35, (512, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, reinterpret_tensor(primals_26, (768, 768), (1, 768), 0), out=buf42)
        buf43 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), out=buf43)
        buf44 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf41, reinterpret_tensor(primals_30, (768, 768), (1, 768), 0), out=buf44)
        buf45 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf42, primals_27, buf45, 393216, grid=grid(393216), stream=stream0)
        del primals_27
        buf46 = reinterpret_tensor(buf42, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf43, primals_29, buf46, 393216, grid=grid(393216), stream=stream0)
        del primals_29
        buf47 = reinterpret_tensor(buf43, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf44, primals_31, buf47, 393216, grid=grid(393216), stream=stream0)
        del primals_31
        # Source Nodes: [], Original ATen: []
        buf48 = aten._scaled_dot_product_efficient_attention(buf45, buf46, buf47, None, True, 0.1, scale=0.125)
        buf49 = buf48[0]
        buf50 = buf48[1]
        buf51 = buf48[2]
        buf52 = buf48[3]
        del buf48
        buf53 = buf44; del buf44  # reuse
        # Source Nodes: [hidden_states_9], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf49, buf53, 393216, grid=grid(393216), stream=stream0)
        buf54 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf53, reinterpret_tensor(primals_32, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf54)
        del primals_33
        # Source Nodes: [hidden_states_10], Original ATen: [aten.native_dropout]
        buf55 = aten.native_dropout(reinterpret_tensor(buf54, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf61 = reinterpret_tensor(buf54, (1, 512, 768), (393216, 768, 1), 0); del buf54  # reuse
        buf62 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf429 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, attention_output_2, hidden_states_12, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf56, buf40, primals_24, primals_25, primals_34, primals_35, buf61, buf62, buf429, 512, 768, grid=grid(512), stream=stream0)
        del primals_25
        buf63 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, buf62, reinterpret_tensor(primals_36, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf63)
        del primals_37
        buf64 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf63, buf64, 1572864, grid=grid(1572864), stream=stream0)
        buf65 = reinterpret_tensor(buf56, (512, 768), (768, 1), 0); del buf56  # reuse
        # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_39, buf64, reinterpret_tensor(primals_38, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf65)
        del primals_39
        # Source Nodes: [hidden_states_15], Original ATen: [aten.native_dropout]
        buf66 = aten.native_dropout(reinterpret_tensor(buf65, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf67 = buf66[0]
        buf68 = buf66[1]
        del buf66
        buf72 = reinterpret_tensor(buf65, (1, 512, 768), (393216, 768, 1), 0); del buf65  # reuse
        buf73 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf428 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_13, attention_output_2, hidden_states_17, mixed_query_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf67, buf61, primals_34, primals_35, primals_40, primals_41, buf72, buf73, buf428, 512, 768, grid=grid(512), stream=stream0)
        del primals_35
        buf74 = reinterpret_tensor(buf67, (512, 768), (768, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf73, reinterpret_tensor(primals_42, (768, 768), (1, 768), 0), out=buf74)
        buf75 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf73, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), out=buf75)
        buf76 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf73, reinterpret_tensor(primals_46, (768, 768), (1, 768), 0), out=buf76)
        buf77 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf74, primals_43, buf77, 393216, grid=grid(393216), stream=stream0)
        del primals_43
        buf78 = reinterpret_tensor(buf74, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf75, primals_45, buf78, 393216, grid=grid(393216), stream=stream0)
        del primals_45
        buf79 = reinterpret_tensor(buf75, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf76, primals_47, buf79, 393216, grid=grid(393216), stream=stream0)
        del primals_47
        # Source Nodes: [], Original ATen: []
        buf80 = aten._scaled_dot_product_efficient_attention(buf77, buf78, buf79, None, True, 0.1, scale=0.125)
        buf81 = buf80[0]
        buf82 = buf80[1]
        buf83 = buf80[2]
        buf84 = buf80[3]
        del buf80
        buf85 = buf76; del buf76  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf81, buf85, 393216, grid=grid(393216), stream=stream0)
        buf86 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, buf85, reinterpret_tensor(primals_48, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf86)
        del primals_49
        # Source Nodes: [hidden_states_19], Original ATen: [aten.native_dropout]
        buf87 = aten.native_dropout(reinterpret_tensor(buf86, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf88 = buf87[0]
        buf89 = buf87[1]
        del buf87
        buf93 = reinterpret_tensor(buf86, (1, 512, 768), (393216, 768, 1), 0); del buf86  # reuse
        buf94 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf427 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, attention_output_4, hidden_states_17, hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf88, buf72, primals_40, primals_41, primals_50, primals_51, buf93, buf94, buf427, 512, 768, grid=grid(512), stream=stream0)
        del primals_41
        buf95 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_53, buf94, reinterpret_tensor(primals_52, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf95)
        del primals_53
        buf96 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_23, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf95, buf96, 1572864, grid=grid(1572864), stream=stream0)
        buf97 = reinterpret_tensor(buf88, (512, 768), (768, 1), 0); del buf88  # reuse
        # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_55, buf96, reinterpret_tensor(primals_54, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf97)
        del primals_55
        # Source Nodes: [hidden_states_24], Original ATen: [aten.native_dropout]
        buf98 = aten.native_dropout(reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf99 = buf98[0]
        buf100 = buf98[1]
        del buf98
        buf104 = reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0); del buf97  # reuse
        buf105 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf426 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_16, attention_output_4, hidden_states_26, mixed_query_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf99, buf93, primals_50, primals_51, primals_56, primals_57, buf104, buf105, buf426, 512, 768, grid=grid(512), stream=stream0)
        del primals_51
        buf106 = reinterpret_tensor(buf99, (512, 768), (768, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf105, reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), out=buf106)
        buf107 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf105, reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), out=buf107)
        buf108 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf105, reinterpret_tensor(primals_62, (768, 768), (1, 768), 0), out=buf108)
        buf109 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf106, primals_59, buf109, 393216, grid=grid(393216), stream=stream0)
        del primals_59
        buf110 = reinterpret_tensor(buf106, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf107, primals_61, buf110, 393216, grid=grid(393216), stream=stream0)
        del primals_61
        buf111 = reinterpret_tensor(buf107, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf108, primals_63, buf111, 393216, grid=grid(393216), stream=stream0)
        del primals_63
        # Source Nodes: [], Original ATen: []
        buf112 = aten._scaled_dot_product_efficient_attention(buf109, buf110, buf111, None, True, 0.1, scale=0.125)
        buf113 = buf112[0]
        buf114 = buf112[1]
        buf115 = buf112[2]
        buf116 = buf112[3]
        del buf112
        buf117 = buf108; del buf108  # reuse
        # Source Nodes: [hidden_states_27], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf113, buf117, 393216, grid=grid(393216), stream=stream0)
        buf118 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf117, reinterpret_tensor(primals_64, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf118)
        del primals_65
        # Source Nodes: [hidden_states_28], Original ATen: [aten.native_dropout]
        buf119 = aten.native_dropout(reinterpret_tensor(buf118, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf120 = buf119[0]
        buf121 = buf119[1]
        del buf119
        buf125 = reinterpret_tensor(buf118, (1, 512, 768), (393216, 768, 1), 0); del buf118  # reuse
        buf126 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf425 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, attention_output_6, hidden_states_26, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf120, buf104, primals_56, primals_57, primals_66, primals_67, buf125, buf126, buf425, 512, 768, grid=grid(512), stream=stream0)
        del primals_57
        buf127 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_69, buf126, reinterpret_tensor(primals_68, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf127)
        del primals_69
        buf128 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf127, buf128, 1572864, grid=grid(1572864), stream=stream0)
        buf129 = reinterpret_tensor(buf120, (512, 768), (768, 1), 0); del buf120  # reuse
        # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_71, buf128, reinterpret_tensor(primals_70, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf129)
        del primals_71
        # Source Nodes: [hidden_states_33], Original ATen: [aten.native_dropout]
        buf130 = aten.native_dropout(reinterpret_tensor(buf129, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf131 = buf130[0]
        buf132 = buf130[1]
        del buf130
        buf136 = reinterpret_tensor(buf129, (1, 512, 768), (393216, 768, 1), 0); del buf129  # reuse
        buf137 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf424 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_19, attention_output_6, hidden_states_35, mixed_query_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf131, buf125, primals_66, primals_67, primals_72, primals_73, buf136, buf137, buf424, 512, 768, grid=grid(512), stream=stream0)
        del primals_67
        buf138 = reinterpret_tensor(buf131, (512, 768), (768, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf137, reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), out=buf138)
        buf139 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf137, reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), out=buf139)
        buf140 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf137, reinterpret_tensor(primals_78, (768, 768), (1, 768), 0), out=buf140)
        buf141 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf138, primals_75, buf141, 393216, grid=grid(393216), stream=stream0)
        del primals_75
        buf142 = reinterpret_tensor(buf138, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf139, primals_77, buf142, 393216, grid=grid(393216), stream=stream0)
        del primals_77
        buf143 = reinterpret_tensor(buf139, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf140, primals_79, buf143, 393216, grid=grid(393216), stream=stream0)
        del primals_79
        # Source Nodes: [], Original ATen: []
        buf144 = aten._scaled_dot_product_efficient_attention(buf141, buf142, buf143, None, True, 0.1, scale=0.125)
        buf145 = buf144[0]
        buf146 = buf144[1]
        buf147 = buf144[2]
        buf148 = buf144[3]
        del buf144
        buf149 = buf140; del buf140  # reuse
        # Source Nodes: [hidden_states_36], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf145, buf149, 393216, grid=grid(393216), stream=stream0)
        buf150 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf149, reinterpret_tensor(primals_80, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf150)
        del primals_81
        # Source Nodes: [hidden_states_37], Original ATen: [aten.native_dropout]
        buf151 = aten.native_dropout(reinterpret_tensor(buf150, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf152 = buf151[0]
        buf153 = buf151[1]
        del buf151
        buf157 = reinterpret_tensor(buf150, (1, 512, 768), (393216, 768, 1), 0); del buf150  # reuse
        buf158 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf423 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, attention_output_8, hidden_states_35, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf152, buf136, primals_72, primals_73, primals_82, primals_83, buf157, buf158, buf423, 512, 768, grid=grid(512), stream=stream0)
        del primals_73
        buf159 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_85, buf158, reinterpret_tensor(primals_84, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf159)
        del primals_85
        buf160 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_41, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf159, buf160, 1572864, grid=grid(1572864), stream=stream0)
        buf161 = reinterpret_tensor(buf152, (512, 768), (768, 1), 0); del buf152  # reuse
        # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_87, buf160, reinterpret_tensor(primals_86, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf161)
        del primals_87
        # Source Nodes: [hidden_states_42], Original ATen: [aten.native_dropout]
        buf162 = aten.native_dropout(reinterpret_tensor(buf161, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        buf168 = reinterpret_tensor(buf161, (1, 512, 768), (393216, 768, 1), 0); del buf161  # reuse
        buf169 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf422 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, attention_output_8, hidden_states_44, mixed_query_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf163, buf157, primals_82, primals_83, primals_88, primals_89, buf168, buf169, buf422, 512, 768, grid=grid(512), stream=stream0)
        del primals_83
        buf170 = reinterpret_tensor(buf163, (512, 768), (768, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf169, reinterpret_tensor(primals_90, (768, 768), (1, 768), 0), out=buf170)
        buf171 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf169, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), out=buf171)
        buf172 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf169, reinterpret_tensor(primals_94, (768, 768), (1, 768), 0), out=buf172)
        buf173 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf170, primals_91, buf173, 393216, grid=grid(393216), stream=stream0)
        del primals_91
        buf174 = reinterpret_tensor(buf170, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf171, primals_93, buf174, 393216, grid=grid(393216), stream=stream0)
        del primals_93
        buf175 = reinterpret_tensor(buf171, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf172, primals_95, buf175, 393216, grid=grid(393216), stream=stream0)
        del primals_95
        # Source Nodes: [], Original ATen: []
        buf176 = aten._scaled_dot_product_efficient_attention(buf173, buf174, buf175, None, True, 0.1, scale=0.125)
        buf177 = buf176[0]
        buf178 = buf176[1]
        buf179 = buf176[2]
        buf180 = buf176[3]
        del buf176
        buf181 = buf172; del buf172  # reuse
        # Source Nodes: [hidden_states_45], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf177, buf181, 393216, grid=grid(393216), stream=stream0)
        buf182 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, buf181, reinterpret_tensor(primals_96, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf182)
        del primals_97
        # Source Nodes: [hidden_states_46], Original ATen: [aten.native_dropout]
        buf183 = aten.native_dropout(reinterpret_tensor(buf182, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf184 = buf183[0]
        buf185 = buf183[1]
        del buf183
        buf189 = reinterpret_tensor(buf182, (1, 512, 768), (393216, 768, 1), 0); del buf182  # reuse
        buf190 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf421 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, attention_output_10, hidden_states_44, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf184, buf168, primals_88, primals_89, primals_98, primals_99, buf189, buf190, buf421, 512, 768, grid=grid(512), stream=stream0)
        del primals_89
        buf191 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_101, buf190, reinterpret_tensor(primals_100, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf191)
        del primals_101
        buf192 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf191, buf192, 1572864, grid=grid(1572864), stream=stream0)
        buf193 = reinterpret_tensor(buf184, (512, 768), (768, 1), 0); del buf184  # reuse
        # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_103, buf192, reinterpret_tensor(primals_102, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf193)
        del primals_103
        # Source Nodes: [hidden_states_51], Original ATen: [aten.native_dropout]
        buf194 = aten.native_dropout(reinterpret_tensor(buf193, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf195 = buf194[0]
        buf196 = buf194[1]
        del buf194
        buf200 = reinterpret_tensor(buf193, (1, 512, 768), (393216, 768, 1), 0); del buf193  # reuse
        buf201 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf420 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, attention_output_10, hidden_states_53, mixed_query_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf195, buf189, primals_98, primals_99, primals_104, primals_105, buf200, buf201, buf420, 512, 768, grid=grid(512), stream=stream0)
        del primals_99
        buf202 = reinterpret_tensor(buf195, (512, 768), (768, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), out=buf202)
        buf203 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_108, (768, 768), (1, 768), 0), out=buf203)
        buf204 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_110, (768, 768), (1, 768), 0), out=buf204)
        buf205 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf202, primals_107, buf205, 393216, grid=grid(393216), stream=stream0)
        del primals_107
        buf206 = reinterpret_tensor(buf202, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf203, primals_109, buf206, 393216, grid=grid(393216), stream=stream0)
        del primals_109
        buf207 = reinterpret_tensor(buf203, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf204, primals_111, buf207, 393216, grid=grid(393216), stream=stream0)
        del primals_111
        # Source Nodes: [], Original ATen: []
        buf208 = aten._scaled_dot_product_efficient_attention(buf205, buf206, buf207, None, True, 0.1, scale=0.125)
        buf209 = buf208[0]
        buf210 = buf208[1]
        buf211 = buf208[2]
        buf212 = buf208[3]
        del buf208
        buf213 = buf204; del buf204  # reuse
        # Source Nodes: [hidden_states_54], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf209, buf213, 393216, grid=grid(393216), stream=stream0)
        buf214 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_113, buf213, reinterpret_tensor(primals_112, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf214)
        del primals_113
        # Source Nodes: [hidden_states_55], Original ATen: [aten.native_dropout]
        buf215 = aten.native_dropout(reinterpret_tensor(buf214, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf216 = buf215[0]
        buf217 = buf215[1]
        del buf215
        buf221 = reinterpret_tensor(buf214, (1, 512, 768), (393216, 768, 1), 0); del buf214  # reuse
        buf222 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf419 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, attention_output_12, hidden_states_53, hidden_states_57], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf216, buf200, primals_104, primals_105, primals_114, primals_115, buf221, buf222, buf419, 512, 768, grid=grid(512), stream=stream0)
        del primals_105
        buf223 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_117, buf222, reinterpret_tensor(primals_116, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf223)
        del primals_117
        buf224 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf223, buf224, 1572864, grid=grid(1572864), stream=stream0)
        buf225 = reinterpret_tensor(buf216, (512, 768), (768, 1), 0); del buf216  # reuse
        # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_119, buf224, reinterpret_tensor(primals_118, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf225)
        del primals_119
        # Source Nodes: [hidden_states_60], Original ATen: [aten.native_dropout]
        buf226 = aten.native_dropout(reinterpret_tensor(buf225, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf227 = buf226[0]
        buf228 = buf226[1]
        del buf226
        buf232 = reinterpret_tensor(buf225, (1, 512, 768), (393216, 768, 1), 0); del buf225  # reuse
        buf233 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf418 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, attention_output_12, hidden_states_62, mixed_query_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf227, buf221, primals_114, primals_115, primals_120, primals_121, buf232, buf233, buf418, 512, 768, grid=grid(512), stream=stream0)
        del primals_115
        buf234 = reinterpret_tensor(buf227, (512, 768), (768, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf233, reinterpret_tensor(primals_122, (768, 768), (1, 768), 0), out=buf234)
        buf235 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf233, reinterpret_tensor(primals_124, (768, 768), (1, 768), 0), out=buf235)
        buf236 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf233, reinterpret_tensor(primals_126, (768, 768), (1, 768), 0), out=buf236)
        buf237 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf234, primals_123, buf237, 393216, grid=grid(393216), stream=stream0)
        del primals_123
        buf238 = reinterpret_tensor(buf234, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf235, primals_125, buf238, 393216, grid=grid(393216), stream=stream0)
        del primals_125
        buf239 = reinterpret_tensor(buf235, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf236, primals_127, buf239, 393216, grid=grid(393216), stream=stream0)
        del primals_127
        # Source Nodes: [], Original ATen: []
        buf240 = aten._scaled_dot_product_efficient_attention(buf237, buf238, buf239, None, True, 0.1, scale=0.125)
        buf241 = buf240[0]
        buf242 = buf240[1]
        buf243 = buf240[2]
        buf244 = buf240[3]
        del buf240
        buf245 = buf236; del buf236  # reuse
        # Source Nodes: [hidden_states_63], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf241, buf245, 393216, grid=grid(393216), stream=stream0)
        buf246 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf245, reinterpret_tensor(primals_128, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf246)
        del primals_129
        # Source Nodes: [hidden_states_64], Original ATen: [aten.native_dropout]
        buf247 = aten.native_dropout(reinterpret_tensor(buf246, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf248 = buf247[0]
        buf249 = buf247[1]
        del buf247
        buf253 = reinterpret_tensor(buf246, (1, 512, 768), (393216, 768, 1), 0); del buf246  # reuse
        buf254 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf417 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, attention_output_14, hidden_states_62, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf248, buf232, primals_120, primals_121, primals_130, primals_131, buf253, buf254, buf417, 512, 768, grid=grid(512), stream=stream0)
        del primals_121
        buf255 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_133, buf254, reinterpret_tensor(primals_132, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf255)
        del primals_133
        buf256 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_68, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf255, buf256, 1572864, grid=grid(1572864), stream=stream0)
        buf257 = reinterpret_tensor(buf248, (512, 768), (768, 1), 0); del buf248  # reuse
        # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_135, buf256, reinterpret_tensor(primals_134, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf257)
        del primals_135
        # Source Nodes: [hidden_states_69], Original ATen: [aten.native_dropout]
        buf258 = aten.native_dropout(reinterpret_tensor(buf257, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf259 = buf258[0]
        buf260 = buf258[1]
        del buf258
        buf264 = reinterpret_tensor(buf257, (1, 512, 768), (393216, 768, 1), 0); del buf257  # reuse
        buf265 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf416 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_31, attention_output_14, hidden_states_71, mixed_query_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf259, buf253, primals_130, primals_131, primals_136, primals_137, buf264, buf265, buf416, 512, 768, grid=grid(512), stream=stream0)
        del primals_131
        buf266 = reinterpret_tensor(buf259, (512, 768), (768, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf265, reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), out=buf266)
        buf267 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf265, reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), out=buf267)
        buf268 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf265, reinterpret_tensor(primals_142, (768, 768), (1, 768), 0), out=buf268)
        buf269 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf266, primals_139, buf269, 393216, grid=grid(393216), stream=stream0)
        del primals_139
        buf270 = reinterpret_tensor(buf266, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf267, primals_141, buf270, 393216, grid=grid(393216), stream=stream0)
        del primals_141
        buf271 = reinterpret_tensor(buf267, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf268, primals_143, buf271, 393216, grid=grid(393216), stream=stream0)
        del primals_143
        # Source Nodes: [], Original ATen: []
        buf272 = aten._scaled_dot_product_efficient_attention(buf269, buf270, buf271, None, True, 0.1, scale=0.125)
        buf273 = buf272[0]
        buf274 = buf272[1]
        buf275 = buf272[2]
        buf276 = buf272[3]
        del buf272
        buf277 = buf268; del buf268  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf273, buf277, 393216, grid=grid(393216), stream=stream0)
        buf278 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, buf277, reinterpret_tensor(primals_144, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf278)
        del primals_145
        # Source Nodes: [hidden_states_73], Original ATen: [aten.native_dropout]
        buf279 = aten.native_dropout(reinterpret_tensor(buf278, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf280 = buf279[0]
        buf281 = buf279[1]
        del buf279
        buf285 = reinterpret_tensor(buf278, (1, 512, 768), (393216, 768, 1), 0); del buf278  # reuse
        buf286 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf415 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, attention_output_16, hidden_states_71, hidden_states_75], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf280, buf264, primals_136, primals_137, primals_146, primals_147, buf285, buf286, buf415, 512, 768, grid=grid(512), stream=stream0)
        del primals_137
        buf287 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_149, buf286, reinterpret_tensor(primals_148, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf287)
        del primals_149
        buf288 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_77, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf287, buf288, 1572864, grid=grid(1572864), stream=stream0)
        buf289 = reinterpret_tensor(buf280, (512, 768), (768, 1), 0); del buf280  # reuse
        # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_151, buf288, reinterpret_tensor(primals_150, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf289)
        del primals_151
        # Source Nodes: [hidden_states_78], Original ATen: [aten.native_dropout]
        buf290 = aten.native_dropout(reinterpret_tensor(buf289, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf291 = buf290[0]
        buf292 = buf290[1]
        del buf290
        buf296 = reinterpret_tensor(buf289, (1, 512, 768), (393216, 768, 1), 0); del buf289  # reuse
        buf297 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf414 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_34, attention_output_16, hidden_states_80, mixed_query_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf291, buf285, primals_146, primals_147, primals_152, primals_153, buf296, buf297, buf414, 512, 768, grid=grid(512), stream=stream0)
        del primals_147
        buf298 = reinterpret_tensor(buf291, (512, 768), (768, 1), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_154, (768, 768), (1, 768), 0), out=buf298)
        buf299 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_156, (768, 768), (1, 768), 0), out=buf299)
        buf300 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_158, (768, 768), (1, 768), 0), out=buf300)
        buf301 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf298, primals_155, buf301, 393216, grid=grid(393216), stream=stream0)
        del primals_155
        buf302 = reinterpret_tensor(buf298, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf299, primals_157, buf302, 393216, grid=grid(393216), stream=stream0)
        del primals_157
        buf303 = reinterpret_tensor(buf299, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf300, primals_159, buf303, 393216, grid=grid(393216), stream=stream0)
        del primals_159
        # Source Nodes: [], Original ATen: []
        buf304 = aten._scaled_dot_product_efficient_attention(buf301, buf302, buf303, None, True, 0.1, scale=0.125)
        buf305 = buf304[0]
        buf306 = buf304[1]
        buf307 = buf304[2]
        buf308 = buf304[3]
        del buf304
        buf309 = buf300; del buf300  # reuse
        # Source Nodes: [hidden_states_81], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf305, buf309, 393216, grid=grid(393216), stream=stream0)
        buf310 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf309, reinterpret_tensor(primals_160, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf310)
        del primals_161
        # Source Nodes: [hidden_states_82], Original ATen: [aten.native_dropout]
        buf311 = aten.native_dropout(reinterpret_tensor(buf310, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf312 = buf311[0]
        buf313 = buf311[1]
        del buf311
        buf317 = reinterpret_tensor(buf310, (1, 512, 768), (393216, 768, 1), 0); del buf310  # reuse
        buf318 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf413 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, attention_output_18, hidden_states_80, hidden_states_84], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf312, buf296, primals_152, primals_153, primals_162, primals_163, buf317, buf318, buf413, 512, 768, grid=grid(512), stream=stream0)
        del primals_153
        buf319 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_165, buf318, reinterpret_tensor(primals_164, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf319)
        del primals_165
        buf320 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_86, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf319, buf320, 1572864, grid=grid(1572864), stream=stream0)
        buf321 = reinterpret_tensor(buf312, (512, 768), (768, 1), 0); del buf312  # reuse
        # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_167, buf320, reinterpret_tensor(primals_166, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf321)
        del primals_167
        # Source Nodes: [hidden_states_87], Original ATen: [aten.native_dropout]
        buf322 = aten.native_dropout(reinterpret_tensor(buf321, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf323 = buf322[0]
        buf324 = buf322[1]
        del buf322
        buf328 = reinterpret_tensor(buf321, (1, 512, 768), (393216, 768, 1), 0); del buf321  # reuse
        buf329 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf412 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_37, attention_output_18, hidden_states_89, mixed_query_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf323, buf317, primals_162, primals_163, primals_168, primals_169, buf328, buf329, buf412, 512, 768, grid=grid(512), stream=stream0)
        del primals_163
        buf330 = reinterpret_tensor(buf323, (512, 768), (768, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf329, reinterpret_tensor(primals_170, (768, 768), (1, 768), 0), out=buf330)
        buf331 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf329, reinterpret_tensor(primals_172, (768, 768), (1, 768), 0), out=buf331)
        buf332 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf329, reinterpret_tensor(primals_174, (768, 768), (1, 768), 0), out=buf332)
        buf333 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf330, primals_171, buf333, 393216, grid=grid(393216), stream=stream0)
        del primals_171
        buf334 = reinterpret_tensor(buf330, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf331, primals_173, buf334, 393216, grid=grid(393216), stream=stream0)
        del primals_173
        buf335 = reinterpret_tensor(buf331, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf331  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf332, primals_175, buf335, 393216, grid=grid(393216), stream=stream0)
        del primals_175
        # Source Nodes: [], Original ATen: []
        buf336 = aten._scaled_dot_product_efficient_attention(buf333, buf334, buf335, None, True, 0.1, scale=0.125)
        buf337 = buf336[0]
        buf338 = buf336[1]
        buf339 = buf336[2]
        buf340 = buf336[3]
        del buf336
        buf341 = buf332; del buf332  # reuse
        # Source Nodes: [hidden_states_90], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf337, buf341, 393216, grid=grid(393216), stream=stream0)
        buf342 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf341, reinterpret_tensor(primals_176, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf342)
        del primals_177
        # Source Nodes: [hidden_states_91], Original ATen: [aten.native_dropout]
        buf343 = aten.native_dropout(reinterpret_tensor(buf342, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf344 = buf343[0]
        buf345 = buf343[1]
        del buf343
        buf349 = reinterpret_tensor(buf342, (1, 512, 768), (393216, 768, 1), 0); del buf342  # reuse
        buf350 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf411 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_39, attention_output_20, hidden_states_89, hidden_states_93], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf344, buf328, primals_168, primals_169, primals_178, primals_179, buf349, buf350, buf411, 512, 768, grid=grid(512), stream=stream0)
        del primals_169
        buf351 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_181, buf350, reinterpret_tensor(primals_180, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf351)
        del primals_181
        buf352 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf351, buf352, 1572864, grid=grid(1572864), stream=stream0)
        buf353 = reinterpret_tensor(buf344, (512, 768), (768, 1), 0); del buf344  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_183, buf352, reinterpret_tensor(primals_182, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf353)
        del primals_183
        # Source Nodes: [hidden_states_96], Original ATen: [aten.native_dropout]
        buf354 = aten.native_dropout(reinterpret_tensor(buf353, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf355 = buf354[0]
        buf356 = buf354[1]
        del buf354
        buf360 = reinterpret_tensor(buf353, (1, 512, 768), (393216, 768, 1), 0); del buf353  # reuse
        buf361 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf410 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_40, attention_output_20, hidden_states_98, mixed_query_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf355, buf349, primals_178, primals_179, primals_184, primals_185, buf360, buf361, buf410, 512, 768, grid=grid(512), stream=stream0)
        del primals_179
        buf362 = reinterpret_tensor(buf355, (512, 768), (768, 1), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf361, reinterpret_tensor(primals_186, (768, 768), (1, 768), 0), out=buf362)
        buf363 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf361, reinterpret_tensor(primals_188, (768, 768), (1, 768), 0), out=buf363)
        buf364 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf361, reinterpret_tensor(primals_190, (768, 768), (1, 768), 0), out=buf364)
        buf365 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf362, primals_187, buf365, 393216, grid=grid(393216), stream=stream0)
        del primals_187
        buf366 = reinterpret_tensor(buf362, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf363, primals_189, buf366, 393216, grid=grid(393216), stream=stream0)
        del primals_189
        buf367 = reinterpret_tensor(buf363, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf364, primals_191, buf367, 393216, grid=grid(393216), stream=stream0)
        del primals_191
        # Source Nodes: [], Original ATen: []
        buf368 = aten._scaled_dot_product_efficient_attention(buf365, buf366, buf367, None, True, 0.1, scale=0.125)
        buf369 = buf368[0]
        buf370 = buf368[1]
        buf371 = buf368[2]
        buf372 = buf368[3]
        del buf368
        buf373 = buf364; del buf364  # reuse
        # Source Nodes: [hidden_states_99], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf369, buf373, 393216, grid=grid(393216), stream=stream0)
        buf374 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_193, buf373, reinterpret_tensor(primals_192, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf374)
        del primals_193
        # Source Nodes: [hidden_states_100], Original ATen: [aten.native_dropout]
        buf375 = aten.native_dropout(reinterpret_tensor(buf374, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf376 = buf375[0]
        buf377 = buf375[1]
        del buf375
        buf381 = reinterpret_tensor(buf374, (1, 512, 768), (393216, 768, 1), 0); del buf374  # reuse
        buf382 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf409 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, attention_output_22, hidden_states_102, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf376, buf360, primals_184, primals_185, primals_194, primals_195, buf381, buf382, buf409, 512, 768, grid=grid(512), stream=stream0)
        del primals_185
        buf383 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_197, buf382, reinterpret_tensor(primals_196, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf383)
        del primals_197
        buf384 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_104, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf383, buf384, 1572864, grid=grid(1572864), stream=stream0)
        buf385 = reinterpret_tensor(buf376, (512, 768), (768, 1), 0); del buf376  # reuse
        # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_199, buf384, reinterpret_tensor(primals_198, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf385)
        del primals_199
        # Source Nodes: [hidden_states_105], Original ATen: [aten.native_dropout]
        buf386 = aten.native_dropout(reinterpret_tensor(buf385, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf387 = buf386[0]
        buf388 = buf386[1]
        del buf386
        buf392 = reinterpret_tensor(buf385, (1, 512, 768), (393216, 768, 1), 0); del buf385  # reuse
        buf393 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf408 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_43, attention_output_22, hidden_states_108, sequence_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf387, buf381, primals_194, primals_195, primals_200, primals_201, buf392, buf393, buf408, 512, 768, grid=grid(512), stream=stream0)
        del primals_195
        del primals_201
        buf394 = reinterpret_tensor(buf387, (512, 768), (768, 1), 0); del buf387  # reuse
        # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_205, buf393, reinterpret_tensor(primals_204, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf394)
        del primals_205
        buf398 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf399 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf407 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, hidden_states_111, prediction_scores], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7.run(buf394, primals_206, primals_207, buf398, buf399, buf407, 512, 768, grid=grid(512), stream=stream0)
        del primals_207
        buf400 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_209, buf399, reinterpret_tensor(primals_208, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf400)
        del primals_209
        buf403 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_8.run(buf400, buf403, 512, 30522, grid=grid(512), stream=stream0)
        buf406 = empty((), device='cuda', dtype=torch.float32)
        buf405 = empty((), device='cuda', dtype=torch.float32)
        buf433 = buf406; del buf406  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_9.run(buf433, primals_212, buf403, buf405, 1, 512, grid=grid(1), stream=stream0)
        buf434 = empty((1, 512, 4), device='cuda', dtype=torch.int64)
        # Source Nodes: [bbox], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_10.run(buf434, 2048, grid=grid(2048), stream=stream0)
        return (buf433, reinterpret_tensor(buf400, (1, 512, 30522), (15627264, 30522, 1), 0), primals_8, primals_18, primals_24, primals_34, primals_40, primals_50, primals_56, primals_66, primals_72, primals_82, primals_88, primals_98, primals_104, primals_114, primals_120, primals_130, primals_136, primals_146, primals_152, primals_162, primals_168, primals_178, primals_184, primals_194, primals_200, primals_206, primals_211, primals_212, buf0, primals_210, reinterpret_tensor(buf434, (1, 512), (0, 4), 0), reinterpret_tensor(buf434, (1, 512), (0, 4), 1), reinterpret_tensor(buf434, (1, 512), (0, 4), 2), reinterpret_tensor(buf434, (1, 512), (0, 4), 3), buf5, buf9, reinterpret_tensor(buf8, (512, 768), (768, 1), 0), buf13, buf14, buf15, buf18, buf19, buf20, buf17, buf21, buf25, buf29, buf30, buf31, buf32, buf36, buf40, buf41, buf45, buf46, buf47, buf50, buf51, buf52, buf49, buf53, buf57, buf61, buf62, buf63, buf64, buf68, buf72, buf73, buf77, buf78, buf79, buf82, buf83, buf84, buf81, buf85, buf89, buf93, buf94, buf95, buf96, buf100, buf104, buf105, buf109, buf110, buf111, buf114, buf115, buf116, buf113, buf117, buf121, buf125, buf126, buf127, buf128, buf132, buf136, buf137, buf141, buf142, buf143, buf146, buf147, buf148, buf145, buf149, buf153, buf157, buf158, buf159, buf160, buf164, buf168, buf169, buf173, buf174, buf175, buf178, buf179, buf180, buf177, buf181, buf185, buf189, buf190, buf191, buf192, buf196, buf200, buf201, buf205, buf206, buf207, buf210, buf211, buf212, buf209, buf213, buf217, buf221, buf222, buf223, buf224, buf228, buf232, buf233, buf237, buf238, buf239, buf242, buf243, buf244, buf241, buf245, buf249, buf253, buf254, buf255, buf256, buf260, buf264, buf265, buf269, buf270, buf271, buf274, buf275, buf276, buf273, buf277, buf281, buf285, buf286, buf287, buf288, buf292, buf296, buf297, buf301, buf302, buf303, buf306, buf307, buf308, buf305, buf309, buf313, buf317, buf318, buf319, buf320, buf324, buf328, buf329, buf333, buf334, buf335, buf338, buf339, buf340, buf337, buf341, buf345, buf349, buf350, buf351, buf352, buf356, buf360, buf361, buf365, buf366, buf367, buf370, buf371, buf372, buf369, buf373, buf377, buf381, buf382, buf383, buf384, buf388, buf392, buf393, buf394, buf398, buf399, buf403, buf405, reinterpret_tensor(primals_208, (30522, 768), (768, 1), 0), buf407, reinterpret_tensor(primals_204, (768, 768), (768, 1), 0), buf408, reinterpret_tensor(primals_198, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_196, (3072, 768), (768, 1), 0), buf409, reinterpret_tensor(primals_192, (768, 768), (768, 1), 0), reinterpret_tensor(primals_190, (768, 768), (768, 1), 0), reinterpret_tensor(primals_188, (768, 768), (768, 1), 0), reinterpret_tensor(primals_186, (768, 768), (768, 1), 0), buf410, reinterpret_tensor(primals_182, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_180, (3072, 768), (768, 1), 0), buf411, reinterpret_tensor(primals_176, (768, 768), (768, 1), 0), reinterpret_tensor(primals_174, (768, 768), (768, 1), 0), reinterpret_tensor(primals_172, (768, 768), (768, 1), 0), reinterpret_tensor(primals_170, (768, 768), (768, 1), 0), buf412, reinterpret_tensor(primals_166, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_164, (3072, 768), (768, 1), 0), buf413, reinterpret_tensor(primals_160, (768, 768), (768, 1), 0), reinterpret_tensor(primals_158, (768, 768), (768, 1), 0), reinterpret_tensor(primals_156, (768, 768), (768, 1), 0), reinterpret_tensor(primals_154, (768, 768), (768, 1), 0), buf414, reinterpret_tensor(primals_150, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_148, (3072, 768), (768, 1), 0), buf415, reinterpret_tensor(primals_144, (768, 768), (768, 1), 0), reinterpret_tensor(primals_142, (768, 768), (768, 1), 0), reinterpret_tensor(primals_140, (768, 768), (768, 1), 0), reinterpret_tensor(primals_138, (768, 768), (768, 1), 0), buf416, reinterpret_tensor(primals_134, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_132, (3072, 768), (768, 1), 0), buf417, reinterpret_tensor(primals_128, (768, 768), (768, 1), 0), reinterpret_tensor(primals_126, (768, 768), (768, 1), 0), reinterpret_tensor(primals_124, (768, 768), (768, 1), 0), reinterpret_tensor(primals_122, (768, 768), (768, 1), 0), buf418, reinterpret_tensor(primals_118, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_116, (3072, 768), (768, 1), 0), buf419, reinterpret_tensor(primals_112, (768, 768), (768, 1), 0), reinterpret_tensor(primals_110, (768, 768), (768, 1), 0), reinterpret_tensor(primals_108, (768, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), buf420, reinterpret_tensor(primals_102, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_100, (3072, 768), (768, 1), 0), buf421, reinterpret_tensor(primals_96, (768, 768), (768, 1), 0), reinterpret_tensor(primals_94, (768, 768), (768, 1), 0), reinterpret_tensor(primals_92, (768, 768), (768, 1), 0), reinterpret_tensor(primals_90, (768, 768), (768, 1), 0), buf422, reinterpret_tensor(primals_86, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_84, (3072, 768), (768, 1), 0), buf423, reinterpret_tensor(primals_80, (768, 768), (768, 1), 0), reinterpret_tensor(primals_78, (768, 768), (768, 1), 0), reinterpret_tensor(primals_76, (768, 768), (768, 1), 0), reinterpret_tensor(primals_74, (768, 768), (768, 1), 0), buf424, reinterpret_tensor(primals_70, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_68, (3072, 768), (768, 1), 0), buf425, reinterpret_tensor(primals_64, (768, 768), (768, 1), 0), reinterpret_tensor(primals_62, (768, 768), (768, 1), 0), reinterpret_tensor(primals_60, (768, 768), (768, 1), 0), reinterpret_tensor(primals_58, (768, 768), (768, 1), 0), buf426, reinterpret_tensor(primals_54, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_52, (3072, 768), (768, 1), 0), buf427, reinterpret_tensor(primals_48, (768, 768), (768, 1), 0), reinterpret_tensor(primals_46, (768, 768), (768, 1), 0), reinterpret_tensor(primals_44, (768, 768), (768, 1), 0), reinterpret_tensor(primals_42, (768, 768), (768, 1), 0), buf428, reinterpret_tensor(primals_38, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_36, (3072, 768), (768, 1), 0), buf429, reinterpret_tensor(primals_32, (768, 768), (768, 1), 0), reinterpret_tensor(primals_30, (768, 768), (768, 1), 0), reinterpret_tensor(primals_28, (768, 768), (768, 1), 0), reinterpret_tensor(primals_26, (768, 768), (768, 1), 0), buf430, reinterpret_tensor(primals_22, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_20, (3072, 768), (768, 1), 0), buf431, reinterpret_tensor(primals_16, (768, 768), (768, 1), 0), reinterpret_tensor(primals_14, (768, 768), (768, 1), 0), reinterpret_tensor(primals_12, (768, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), buf432, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_212 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForMaskedLM', benchmark_compiled_module)
