
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


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqvxxmfvhtsblcyvx5s2y2mkckhjqnrfalu5turf5e3hhxzu2kv.py
# Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
# cumsum => cumsum
# mask => convert_element_type
# ne => ne
triton_poi_fused__to_copy_cumsum_ne_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/75/c75ft2k7hloosrzwecmszqcnoelcrxoiwjbcclds6atikfuzwu2s.py
# Source Nodes: [add, embeddings, embeddings_1, embeddings_2, incremental_indices, inputs_embeds, long, mask, ne, position_embeddings, position_ids, token_type_embeddings, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.ne]
# add => add
# embeddings => add_2
# embeddings_1 => add_3
# embeddings_2 => add_4, add_5, mul_2, mul_3, rsqrt, sub_1, var_mean
# incremental_indices => mul_1
# inputs_embeds => embedding
# long => convert_element_type_2
# mask => convert_element_type
# ne => ne
# position_embeddings => embedding_2
# position_ids => add_1
# token_type_embeddings => embedding_1
# type_as => convert_element_type_1
triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_native_layer_norm_backward_ne_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_native_layer_norm_backward_ne_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 != tmp5
    tmp7 = tmp6.to(tl.int32)
    tmp8 = tmp3 * tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9 + tmp5
    tmp11 = tmp4 + 50265
    tmp12 = tmp4 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp4)
    tl.device_assert(((0 <= tmp13) & (tmp13 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp13 < 50265")
    tmp14 = tl.load(in_ptr1 + (r1 + (768*tmp13)), rmask & xmask, other=0.0)
    tmp16 = tmp15 + 2
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tl.device_assert(((0 <= tmp18) & (tmp18 < 2)) | ~xmask, "index out of bounds: 0 <= tmp18 < 2")
    tmp19 = tl.load(in_ptr3 + (r1 + (768*tmp18)), rmask & xmask, other=0.0)
    tmp20 = tmp14 + tmp19
    tmp21 = tmp10 + 512
    tmp22 = tmp10 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp10)
    tl.device_assert(((0 <= tmp23) & (tmp23 < 512)) | ~xmask, "index out of bounds: 0 <= tmp23 < 512")
    tmp24 = tl.load(in_ptr4 + (r1 + (768*tmp23)), rmask & xmask, other=0.0)
    tmp25 = tmp20 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tl.full([1], 768, tl.int32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 / tmp34
    tmp36 = tmp26 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp42 = tmp25 - tmp35
    tmp43 = 768.0
    tmp44 = tmp41 / tmp43
    tmp45 = 1e-12
    tmp46 = tmp44 + tmp45
    tmp47 = tl.math.rsqrt(tmp46)
    tmp48 = tmp42 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tmp47 / tmp43
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp48, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp52, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp53, xmask)
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
# Source Nodes: [add_4, attention_output, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_4 => add_7
# attention_output => add_8, add_9, mul_4, mul_5, rsqrt_1, sub_3, var_mean_1
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
# intermediate_output => add_10, erf, mul_6, mul_7, mul_8
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
# Source Nodes: [add_5, attention_output, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_5 => add_11
# attention_output => add_9, mul_5
# hidden_states_8 => add_12, add_13, mul_10, mul_9, rsqrt_2, sub_4, var_mean_2
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
# Source Nodes: [prediction_scores, x_37, x_38], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# prediction_scores => view_266
# x_37 => add_102, erf_12, mul_88, mul_89, mul_90
# x_38 => add_103, add_104, mul_91, mul_92, rsqrt_25, sub_38, var_mean_25
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


# kernel path: /tmp/torchinductor_youkaichao/qs/cqsun7ufkbbsruvft3db6uhjp24rsjrpi24f76z24guifou57t2k.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_12, exp_12, log, sub_39, sub_40, sum_13
triton_red_fused__log_softmax_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 511
    rnumel = 50265
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
        tmp0 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp10 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.log(tmp8)
        tmp13 = tmp11 - tmp12
        tl.store(out_ptr2 + (r1 + (50265*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwaccdm4r3mlmgievgg2m3t3n273chd7wl3pprsvfpznxmbfqstg.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => convert_element_type_3, div_24, full_default_1, full_default_2, ne_1, neg, sum_14, sum_15, where_1
triton_per_fused_nll_loss_backward_nll_loss_forward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*i64', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_backward_nll_loss_forward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.where(tmp2, tmp0, tmp8)
    tmp10 = tmp9 + 50265
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 50265), "index out of bounds: 0 <= tmp12 < 50265")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (50265*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp7.to(tl.float32)
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp2, rmask)
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp9, rmask)
    tl.store(out_ptr3 + (tl.full([1], 0, tl.int32)), tmp21, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp22, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206 = args
    args.clear()
    assert_size_stride(primals_1, (50265, 768), (768, 1))
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
    assert_size_stride(primals_198, (768, 768), (768, 1))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (50265, 768), (768, 1))
    assert_size_stride(primals_203, (50265, ), (1, ))
    assert_size_stride(primals_204, (1, 512), (512, 1))
    assert_size_stride(primals_205, (1, 512), (512, 1))
    assert_size_stride(primals_206, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512), device='cuda', dtype=torch.int32)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__to_copy_cumsum_ne_0.run(primals_206, buf0, 512, grid=grid(512), stream=stream0)
        # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
        buf1 = aten.cumsum(buf0, 1)
        del buf0
        buf2 = buf1
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf8 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf9 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf437 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, embeddings, embeddings_1, embeddings_2, incremental_indices, inputs_embeds, long, mask, ne, position_embeddings, position_ids, token_type_embeddings, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.ne]
        triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_native_layer_norm_backward_ne_1.run(buf3, primals_206, primals_1, primals_204, primals_2, primals_3, primals_4, primals_5, buf4, buf8, buf9, buf437, 512, 768, grid=grid(512), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        # Source Nodes: [embedding_output, embeddings_2], Original ATen: [aten.native_dropout, aten.native_layer_norm]
        buf10 = aten.native_dropout(buf9, 0.1, True)
        buf11 = buf10[0]
        buf12 = buf10[1]
        del buf10
        buf13 = reinterpret_tensor(buf9, (512, 768), (768, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (512, 768), (768, 1), 0), reinterpret_tensor(primals_6, (768, 768), (1, 768), 0), out=buf13)
        buf14 = reinterpret_tensor(buf4, (512, 768), (768, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (512, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf14)
        buf15 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (512, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf15)
        buf16 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf13, primals_7, buf16, 393216, grid=grid(393216), stream=stream0)
        del primals_7
        buf17 = reinterpret_tensor(buf13, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf14, primals_9, buf17, 393216, grid=grid(393216), stream=stream0)
        del primals_9
        buf18 = reinterpret_tensor(buf14, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf15, primals_11, buf18, 393216, grid=grid(393216), stream=stream0)
        del primals_11
        # Source Nodes: [], Original ATen: []
        buf19 = aten._scaled_dot_product_efficient_attention(buf16, buf17, buf18, None, True, 0.1, scale=0.125)
        buf20 = buf19[0]
        buf21 = buf19[1]
        buf22 = buf19[2]
        buf23 = buf19[3]
        del buf19
        buf24 = buf15; del buf15  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf20, buf24, 393216, grid=grid(393216), stream=stream0)
        buf25 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, buf24, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del primals_13
        # Source Nodes: [hidden_states_1], Original ATen: [aten.native_dropout]
        buf26 = aten.native_dropout(reinterpret_tensor(buf25, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf32 = reinterpret_tensor(buf25, (1, 512, 768), (393216, 768, 1), 0); del buf25  # reuse
        buf33 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf436 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_4, attention_output, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4.run(buf27, buf11, primals_14, primals_15, buf32, buf33, buf436, 512, 768, grid=grid(512), stream=stream0)
        buf34 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, buf33, reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf34)
        del primals_17
        buf35 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf34, buf35, 1572864, grid=grid(1572864), stream=stream0)
        buf36 = reinterpret_tensor(buf27, (512, 768), (768, 1), 0); del buf27  # reuse
        # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf35, reinterpret_tensor(primals_18, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf36)
        del primals_19
        # Source Nodes: [hidden_states_6], Original ATen: [aten.native_dropout]
        buf37 = aten.native_dropout(reinterpret_tensor(buf36, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf38 = buf37[0]
        buf39 = buf37[1]
        del buf37
        buf43 = reinterpret_tensor(buf36, (1, 512, 768), (393216, 768, 1), 0); del buf36  # reuse
        buf44 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf435 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, attention_output, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf38, buf32, primals_14, primals_15, primals_20, primals_21, buf43, buf44, buf435, 512, 768, grid=grid(512), stream=stream0)
        del primals_15
        buf45 = reinterpret_tensor(buf38, (512, 768), (768, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf44, reinterpret_tensor(primals_22, (768, 768), (1, 768), 0), out=buf45)
        buf46 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf44, reinterpret_tensor(primals_24, (768, 768), (1, 768), 0), out=buf46)
        buf47 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf44, reinterpret_tensor(primals_26, (768, 768), (1, 768), 0), out=buf47)
        buf48 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf45, primals_23, buf48, 393216, grid=grid(393216), stream=stream0)
        del primals_23
        buf49 = reinterpret_tensor(buf45, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf46, primals_25, buf49, 393216, grid=grid(393216), stream=stream0)
        del primals_25
        buf50 = reinterpret_tensor(buf46, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf47, primals_27, buf50, 393216, grid=grid(393216), stream=stream0)
        del primals_27
        # Source Nodes: [], Original ATen: []
        buf51 = aten._scaled_dot_product_efficient_attention(buf48, buf49, buf50, None, True, 0.1, scale=0.125)
        buf52 = buf51[0]
        buf53 = buf51[1]
        buf54 = buf51[2]
        buf55 = buf51[3]
        del buf51
        buf56 = buf47; del buf47  # reuse
        # Source Nodes: [hidden_states_9], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf52, buf56, 393216, grid=grid(393216), stream=stream0)
        buf57 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, buf56, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf57)
        del primals_29
        # Source Nodes: [hidden_states_10], Original ATen: [aten.native_dropout]
        buf58 = aten.native_dropout(reinterpret_tensor(buf57, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf59 = buf58[0]
        buf60 = buf58[1]
        del buf58
        buf64 = reinterpret_tensor(buf57, (1, 512, 768), (393216, 768, 1), 0); del buf57  # reuse
        buf65 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf434 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_7, attention_output_2, hidden_states_12, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf59, buf43, primals_20, primals_21, primals_30, primals_31, buf64, buf65, buf434, 512, 768, grid=grid(512), stream=stream0)
        del primals_21
        buf66 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf65, reinterpret_tensor(primals_32, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf66)
        del primals_33
        buf67 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf66, buf67, 1572864, grid=grid(1572864), stream=stream0)
        buf68 = reinterpret_tensor(buf59, (512, 768), (768, 1), 0); del buf59  # reuse
        # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, buf67, reinterpret_tensor(primals_34, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf68)
        del primals_35
        # Source Nodes: [hidden_states_15], Original ATen: [aten.native_dropout]
        buf69 = aten.native_dropout(reinterpret_tensor(buf68, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf75 = reinterpret_tensor(buf68, (1, 512, 768), (393216, 768, 1), 0); del buf68  # reuse
        buf76 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf433 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, attention_output_2, hidden_states_17, mixed_query_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf70, buf64, primals_30, primals_31, primals_36, primals_37, buf75, buf76, buf433, 512, 768, grid=grid(512), stream=stream0)
        del primals_31
        buf77 = reinterpret_tensor(buf70, (512, 768), (768, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf76, reinterpret_tensor(primals_38, (768, 768), (1, 768), 0), out=buf77)
        buf78 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf76, reinterpret_tensor(primals_40, (768, 768), (1, 768), 0), out=buf78)
        buf79 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf76, reinterpret_tensor(primals_42, (768, 768), (1, 768), 0), out=buf79)
        buf80 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf77, primals_39, buf80, 393216, grid=grid(393216), stream=stream0)
        del primals_39
        buf81 = reinterpret_tensor(buf77, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf78, primals_41, buf81, 393216, grid=grid(393216), stream=stream0)
        del primals_41
        buf82 = reinterpret_tensor(buf78, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf79, primals_43, buf82, 393216, grid=grid(393216), stream=stream0)
        del primals_43
        # Source Nodes: [], Original ATen: []
        buf83 = aten._scaled_dot_product_efficient_attention(buf80, buf81, buf82, None, True, 0.1, scale=0.125)
        buf84 = buf83[0]
        buf85 = buf83[1]
        buf86 = buf83[2]
        buf87 = buf83[3]
        del buf83
        buf88 = buf79; del buf79  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf84, buf88, 393216, grid=grid(393216), stream=stream0)
        buf89 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, buf88, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf89)
        del primals_45
        # Source Nodes: [hidden_states_19], Original ATen: [aten.native_dropout]
        buf90 = aten.native_dropout(reinterpret_tensor(buf89, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf91 = buf90[0]
        buf92 = buf90[1]
        del buf90
        buf96 = reinterpret_tensor(buf89, (1, 512, 768), (393216, 768, 1), 0); del buf89  # reuse
        buf97 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf432 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, attention_output_4, hidden_states_17, hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf91, buf75, primals_36, primals_37, primals_46, primals_47, buf96, buf97, buf432, 512, 768, grid=grid(512), stream=stream0)
        del primals_37
        buf98 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, buf97, reinterpret_tensor(primals_48, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf98)
        del primals_49
        buf99 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_23, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf98, buf99, 1572864, grid=grid(1572864), stream=stream0)
        buf100 = reinterpret_tensor(buf91, (512, 768), (768, 1), 0); del buf91  # reuse
        # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_51, buf99, reinterpret_tensor(primals_50, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf100)
        del primals_51
        # Source Nodes: [hidden_states_24], Original ATen: [aten.native_dropout]
        buf101 = aten.native_dropout(reinterpret_tensor(buf100, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf102 = buf101[0]
        buf103 = buf101[1]
        del buf101
        buf107 = reinterpret_tensor(buf100, (1, 512, 768), (393216, 768, 1), 0); del buf100  # reuse
        buf108 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf431 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, attention_output_4, hidden_states_26, mixed_query_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf102, buf96, primals_46, primals_47, primals_52, primals_53, buf107, buf108, buf431, 512, 768, grid=grid(512), stream=stream0)
        del primals_47
        buf109 = reinterpret_tensor(buf102, (512, 768), (768, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf108, reinterpret_tensor(primals_54, (768, 768), (1, 768), 0), out=buf109)
        buf110 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf108, reinterpret_tensor(primals_56, (768, 768), (1, 768), 0), out=buf110)
        buf111 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf108, reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), out=buf111)
        buf112 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf109, primals_55, buf112, 393216, grid=grid(393216), stream=stream0)
        del primals_55
        buf113 = reinterpret_tensor(buf109, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf110, primals_57, buf113, 393216, grid=grid(393216), stream=stream0)
        del primals_57
        buf114 = reinterpret_tensor(buf110, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf111, primals_59, buf114, 393216, grid=grid(393216), stream=stream0)
        del primals_59
        # Source Nodes: [], Original ATen: []
        buf115 = aten._scaled_dot_product_efficient_attention(buf112, buf113, buf114, None, True, 0.1, scale=0.125)
        buf116 = buf115[0]
        buf117 = buf115[1]
        buf118 = buf115[2]
        buf119 = buf115[3]
        del buf115
        buf120 = buf111; del buf111  # reuse
        # Source Nodes: [hidden_states_27], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf116, buf120, 393216, grid=grid(393216), stream=stream0)
        buf121 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_61, buf120, reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf121)
        del primals_61
        # Source Nodes: [hidden_states_28], Original ATen: [aten.native_dropout]
        buf122 = aten.native_dropout(reinterpret_tensor(buf121, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf128 = reinterpret_tensor(buf121, (1, 512, 768), (393216, 768, 1), 0); del buf121  # reuse
        buf129 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf430 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_13, attention_output_6, hidden_states_26, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf123, buf107, primals_52, primals_53, primals_62, primals_63, buf128, buf129, buf430, 512, 768, grid=grid(512), stream=stream0)
        del primals_53
        buf130 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf129, reinterpret_tensor(primals_64, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf130)
        del primals_65
        buf131 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf130, buf131, 1572864, grid=grid(1572864), stream=stream0)
        buf132 = reinterpret_tensor(buf123, (512, 768), (768, 1), 0); del buf123  # reuse
        # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_67, buf131, reinterpret_tensor(primals_66, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf132)
        del primals_67
        # Source Nodes: [hidden_states_33], Original ATen: [aten.native_dropout]
        buf133 = aten.native_dropout(reinterpret_tensor(buf132, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf134 = buf133[0]
        buf135 = buf133[1]
        del buf133
        buf139 = reinterpret_tensor(buf132, (1, 512, 768), (393216, 768, 1), 0); del buf132  # reuse
        buf140 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf429 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, attention_output_6, hidden_states_35, mixed_query_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf134, buf128, primals_62, primals_63, primals_68, primals_69, buf139, buf140, buf429, 512, 768, grid=grid(512), stream=stream0)
        del primals_63
        buf141 = reinterpret_tensor(buf134, (512, 768), (768, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf140, reinterpret_tensor(primals_70, (768, 768), (1, 768), 0), out=buf141)
        buf142 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf140, reinterpret_tensor(primals_72, (768, 768), (1, 768), 0), out=buf142)
        buf143 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf140, reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), out=buf143)
        buf144 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf141, primals_71, buf144, 393216, grid=grid(393216), stream=stream0)
        del primals_71
        buf145 = reinterpret_tensor(buf141, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf142, primals_73, buf145, 393216, grid=grid(393216), stream=stream0)
        del primals_73
        buf146 = reinterpret_tensor(buf142, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf143, primals_75, buf146, 393216, grid=grid(393216), stream=stream0)
        del primals_75
        # Source Nodes: [], Original ATen: []
        buf147 = aten._scaled_dot_product_efficient_attention(buf144, buf145, buf146, None, True, 0.1, scale=0.125)
        buf148 = buf147[0]
        buf149 = buf147[1]
        buf150 = buf147[2]
        buf151 = buf147[3]
        del buf147
        buf152 = buf143; del buf143  # reuse
        # Source Nodes: [hidden_states_36], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf148, buf152, 393216, grid=grid(393216), stream=stream0)
        buf153 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, buf152, reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf153)
        del primals_77
        # Source Nodes: [hidden_states_37], Original ATen: [aten.native_dropout]
        buf154 = aten.native_dropout(reinterpret_tensor(buf153, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf155 = buf154[0]
        buf156 = buf154[1]
        del buf154
        buf160 = reinterpret_tensor(buf153, (1, 512, 768), (393216, 768, 1), 0); del buf153  # reuse
        buf161 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf428 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_16, attention_output_8, hidden_states_35, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf155, buf139, primals_68, primals_69, primals_78, primals_79, buf160, buf161, buf428, 512, 768, grid=grid(512), stream=stream0)
        del primals_69
        buf162 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf161, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf162)
        del primals_81
        buf163 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_41, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf162, buf163, 1572864, grid=grid(1572864), stream=stream0)
        buf164 = reinterpret_tensor(buf155, (512, 768), (768, 1), 0); del buf155  # reuse
        # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_83, buf163, reinterpret_tensor(primals_82, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf164)
        del primals_83
        # Source Nodes: [hidden_states_42], Original ATen: [aten.native_dropout]
        buf165 = aten.native_dropout(reinterpret_tensor(buf164, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf166 = buf165[0]
        buf167 = buf165[1]
        del buf165
        buf171 = reinterpret_tensor(buf164, (1, 512, 768), (393216, 768, 1), 0); del buf164  # reuse
        buf172 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf427 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, attention_output_8, hidden_states_44, mixed_query_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf166, buf160, primals_78, primals_79, primals_84, primals_85, buf171, buf172, buf427, 512, 768, grid=grid(512), stream=stream0)
        del primals_79
        buf173 = reinterpret_tensor(buf166, (512, 768), (768, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf172, reinterpret_tensor(primals_86, (768, 768), (1, 768), 0), out=buf173)
        buf174 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf172, reinterpret_tensor(primals_88, (768, 768), (1, 768), 0), out=buf174)
        buf175 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf172, reinterpret_tensor(primals_90, (768, 768), (1, 768), 0), out=buf175)
        buf176 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf173, primals_87, buf176, 393216, grid=grid(393216), stream=stream0)
        del primals_87
        buf177 = reinterpret_tensor(buf173, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf174, primals_89, buf177, 393216, grid=grid(393216), stream=stream0)
        del primals_89
        buf178 = reinterpret_tensor(buf174, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf175, primals_91, buf178, 393216, grid=grid(393216), stream=stream0)
        del primals_91
        # Source Nodes: [], Original ATen: []
        buf179 = aten._scaled_dot_product_efficient_attention(buf176, buf177, buf178, None, True, 0.1, scale=0.125)
        buf180 = buf179[0]
        buf181 = buf179[1]
        buf182 = buf179[2]
        buf183 = buf179[3]
        del buf179
        buf184 = buf175; del buf175  # reuse
        # Source Nodes: [hidden_states_45], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf180, buf184, 393216, grid=grid(393216), stream=stream0)
        buf185 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, buf184, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf185)
        del primals_93
        # Source Nodes: [hidden_states_46], Original ATen: [aten.native_dropout]
        buf186 = aten.native_dropout(reinterpret_tensor(buf185, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf192 = reinterpret_tensor(buf185, (1, 512, 768), (393216, 768, 1), 0); del buf185  # reuse
        buf193 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf426 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_19, attention_output_10, hidden_states_44, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf187, buf171, primals_84, primals_85, primals_94, primals_95, buf192, buf193, buf426, 512, 768, grid=grid(512), stream=stream0)
        del primals_85
        buf194 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, buf193, reinterpret_tensor(primals_96, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf194)
        del primals_97
        buf195 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf194, buf195, 1572864, grid=grid(1572864), stream=stream0)
        buf196 = reinterpret_tensor(buf187, (512, 768), (768, 1), 0); del buf187  # reuse
        # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_99, buf195, reinterpret_tensor(primals_98, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf196)
        del primals_99
        # Source Nodes: [hidden_states_51], Original ATen: [aten.native_dropout]
        buf197 = aten.native_dropout(reinterpret_tensor(buf196, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf198 = buf197[0]
        buf199 = buf197[1]
        del buf197
        buf203 = reinterpret_tensor(buf196, (1, 512, 768), (393216, 768, 1), 0); del buf196  # reuse
        buf204 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf425 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, attention_output_10, hidden_states_53, mixed_query_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf198, buf192, primals_94, primals_95, primals_100, primals_101, buf203, buf204, buf425, 512, 768, grid=grid(512), stream=stream0)
        del primals_95
        buf205 = reinterpret_tensor(buf198, (512, 768), (768, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf204, reinterpret_tensor(primals_102, (768, 768), (1, 768), 0), out=buf205)
        buf206 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf204, reinterpret_tensor(primals_104, (768, 768), (1, 768), 0), out=buf206)
        buf207 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf204, reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), out=buf207)
        buf208 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf205, primals_103, buf208, 393216, grid=grid(393216), stream=stream0)
        del primals_103
        buf209 = reinterpret_tensor(buf205, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf206, primals_105, buf209, 393216, grid=grid(393216), stream=stream0)
        del primals_105
        buf210 = reinterpret_tensor(buf206, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf207, primals_107, buf210, 393216, grid=grid(393216), stream=stream0)
        del primals_107
        # Source Nodes: [], Original ATen: []
        buf211 = aten._scaled_dot_product_efficient_attention(buf208, buf209, buf210, None, True, 0.1, scale=0.125)
        buf212 = buf211[0]
        buf213 = buf211[1]
        buf214 = buf211[2]
        buf215 = buf211[3]
        del buf211
        buf216 = buf207; del buf207  # reuse
        # Source Nodes: [hidden_states_54], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf212, buf216, 393216, grid=grid(393216), stream=stream0)
        buf217 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_109, buf216, reinterpret_tensor(primals_108, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf217)
        del primals_109
        # Source Nodes: [hidden_states_55], Original ATen: [aten.native_dropout]
        buf218 = aten.native_dropout(reinterpret_tensor(buf217, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf219 = buf218[0]
        buf220 = buf218[1]
        del buf218
        buf224 = reinterpret_tensor(buf217, (1, 512, 768), (393216, 768, 1), 0); del buf217  # reuse
        buf225 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf424 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, attention_output_12, hidden_states_53, hidden_states_57], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf219, buf203, primals_100, primals_101, primals_110, primals_111, buf224, buf225, buf424, 512, 768, grid=grid(512), stream=stream0)
        del primals_101
        buf226 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_113, buf225, reinterpret_tensor(primals_112, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf226)
        del primals_113
        buf227 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf226, buf227, 1572864, grid=grid(1572864), stream=stream0)
        buf228 = reinterpret_tensor(buf219, (512, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_115, buf227, reinterpret_tensor(primals_114, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf228)
        del primals_115
        # Source Nodes: [hidden_states_60], Original ATen: [aten.native_dropout]
        buf229 = aten.native_dropout(reinterpret_tensor(buf228, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf235 = reinterpret_tensor(buf228, (1, 512, 768), (393216, 768, 1), 0); del buf228  # reuse
        buf236 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf423 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, attention_output_12, hidden_states_62, mixed_query_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf230, buf224, primals_110, primals_111, primals_116, primals_117, buf235, buf236, buf423, 512, 768, grid=grid(512), stream=stream0)
        del primals_111
        buf237 = reinterpret_tensor(buf230, (512, 768), (768, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), out=buf237)
        buf238 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_120, (768, 768), (1, 768), 0), out=buf238)
        buf239 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_122, (768, 768), (1, 768), 0), out=buf239)
        buf240 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf237, primals_119, buf240, 393216, grid=grid(393216), stream=stream0)
        del primals_119
        buf241 = reinterpret_tensor(buf237, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf238, primals_121, buf241, 393216, grid=grid(393216), stream=stream0)
        del primals_121
        buf242 = reinterpret_tensor(buf238, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf239, primals_123, buf242, 393216, grid=grid(393216), stream=stream0)
        del primals_123
        # Source Nodes: [], Original ATen: []
        buf243 = aten._scaled_dot_product_efficient_attention(buf240, buf241, buf242, None, True, 0.1, scale=0.125)
        buf244 = buf243[0]
        buf245 = buf243[1]
        buf246 = buf243[2]
        buf247 = buf243[3]
        del buf243
        buf248 = buf239; del buf239  # reuse
        # Source Nodes: [hidden_states_63], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf244, buf248, 393216, grid=grid(393216), stream=stream0)
        buf249 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_125, buf248, reinterpret_tensor(primals_124, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf249)
        del primals_125
        # Source Nodes: [hidden_states_64], Original ATen: [aten.native_dropout]
        buf250 = aten.native_dropout(reinterpret_tensor(buf249, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf251 = buf250[0]
        buf252 = buf250[1]
        del buf250
        buf256 = reinterpret_tensor(buf249, (1, 512, 768), (393216, 768, 1), 0); del buf249  # reuse
        buf257 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf422 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, attention_output_14, hidden_states_62, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf251, buf235, primals_116, primals_117, primals_126, primals_127, buf256, buf257, buf422, 512, 768, grid=grid(512), stream=stream0)
        del primals_117
        buf258 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf257, reinterpret_tensor(primals_128, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf258)
        del primals_129
        buf259 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_68, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf258, buf259, 1572864, grid=grid(1572864), stream=stream0)
        buf260 = reinterpret_tensor(buf251, (512, 768), (768, 1), 0); del buf251  # reuse
        # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_131, buf259, reinterpret_tensor(primals_130, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf260)
        del primals_131
        # Source Nodes: [hidden_states_69], Original ATen: [aten.native_dropout]
        buf261 = aten.native_dropout(reinterpret_tensor(buf260, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf262 = buf261[0]
        buf263 = buf261[1]
        del buf261
        buf267 = reinterpret_tensor(buf260, (1, 512, 768), (393216, 768, 1), 0); del buf260  # reuse
        buf268 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf421 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, attention_output_14, hidden_states_71, mixed_query_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf262, buf256, primals_126, primals_127, primals_132, primals_133, buf267, buf268, buf421, 512, 768, grid=grid(512), stream=stream0)
        del primals_127
        buf269 = reinterpret_tensor(buf262, (512, 768), (768, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf268, reinterpret_tensor(primals_134, (768, 768), (1, 768), 0), out=buf269)
        buf270 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf268, reinterpret_tensor(primals_136, (768, 768), (1, 768), 0), out=buf270)
        buf271 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf268, reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), out=buf271)
        buf272 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf269, primals_135, buf272, 393216, grid=grid(393216), stream=stream0)
        del primals_135
        buf273 = reinterpret_tensor(buf269, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf270, primals_137, buf273, 393216, grid=grid(393216), stream=stream0)
        del primals_137
        buf274 = reinterpret_tensor(buf270, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf271, primals_139, buf274, 393216, grid=grid(393216), stream=stream0)
        del primals_139
        # Source Nodes: [], Original ATen: []
        buf275 = aten._scaled_dot_product_efficient_attention(buf272, buf273, buf274, None, True, 0.1, scale=0.125)
        buf276 = buf275[0]
        buf277 = buf275[1]
        buf278 = buf275[2]
        buf279 = buf275[3]
        del buf275
        buf280 = buf271; del buf271  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf276, buf280, 393216, grid=grid(393216), stream=stream0)
        buf281 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_141, buf280, reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf281)
        del primals_141
        # Source Nodes: [hidden_states_73], Original ATen: [aten.native_dropout]
        buf282 = aten.native_dropout(reinterpret_tensor(buf281, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf283 = buf282[0]
        buf284 = buf282[1]
        del buf282
        buf288 = reinterpret_tensor(buf281, (1, 512, 768), (393216, 768, 1), 0); del buf281  # reuse
        buf289 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf420 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, attention_output_16, hidden_states_71, hidden_states_75], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf283, buf267, primals_132, primals_133, primals_142, primals_143, buf288, buf289, buf420, 512, 768, grid=grid(512), stream=stream0)
        del primals_133
        buf290 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, buf289, reinterpret_tensor(primals_144, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf290)
        del primals_145
        buf291 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_77, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf290, buf291, 1572864, grid=grid(1572864), stream=stream0)
        buf292 = reinterpret_tensor(buf283, (512, 768), (768, 1), 0); del buf283  # reuse
        # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_147, buf291, reinterpret_tensor(primals_146, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf292)
        del primals_147
        # Source Nodes: [hidden_states_78], Original ATen: [aten.native_dropout]
        buf293 = aten.native_dropout(reinterpret_tensor(buf292, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf294 = buf293[0]
        buf295 = buf293[1]
        del buf293
        buf299 = reinterpret_tensor(buf292, (1, 512, 768), (393216, 768, 1), 0); del buf292  # reuse
        buf300 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf419 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, attention_output_16, hidden_states_80, mixed_query_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf294, buf288, primals_142, primals_143, primals_148, primals_149, buf299, buf300, buf419, 512, 768, grid=grid(512), stream=stream0)
        del primals_143
        buf301 = reinterpret_tensor(buf294, (512, 768), (768, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf300, reinterpret_tensor(primals_150, (768, 768), (1, 768), 0), out=buf301)
        buf302 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf300, reinterpret_tensor(primals_152, (768, 768), (1, 768), 0), out=buf302)
        buf303 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf300, reinterpret_tensor(primals_154, (768, 768), (1, 768), 0), out=buf303)
        buf304 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf301, primals_151, buf304, 393216, grid=grid(393216), stream=stream0)
        del primals_151
        buf305 = reinterpret_tensor(buf301, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf302, primals_153, buf305, 393216, grid=grid(393216), stream=stream0)
        del primals_153
        buf306 = reinterpret_tensor(buf302, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf303, primals_155, buf306, 393216, grid=grid(393216), stream=stream0)
        del primals_155
        # Source Nodes: [], Original ATen: []
        buf307 = aten._scaled_dot_product_efficient_attention(buf304, buf305, buf306, None, True, 0.1, scale=0.125)
        buf308 = buf307[0]
        buf309 = buf307[1]
        buf310 = buf307[2]
        buf311 = buf307[3]
        del buf307
        buf312 = buf303; del buf303  # reuse
        # Source Nodes: [hidden_states_81], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf308, buf312, 393216, grid=grid(393216), stream=stream0)
        buf313 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_157, buf312, reinterpret_tensor(primals_156, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf313)
        del primals_157
        # Source Nodes: [hidden_states_82], Original ATen: [aten.native_dropout]
        buf314 = aten.native_dropout(reinterpret_tensor(buf313, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf315 = buf314[0]
        buf316 = buf314[1]
        del buf314
        buf320 = reinterpret_tensor(buf313, (1, 512, 768), (393216, 768, 1), 0); del buf313  # reuse
        buf321 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf418 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_31, attention_output_18, hidden_states_80, hidden_states_84], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf315, buf299, primals_148, primals_149, primals_158, primals_159, buf320, buf321, buf418, 512, 768, grid=grid(512), stream=stream0)
        del primals_149
        buf322 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf321, reinterpret_tensor(primals_160, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf322)
        del primals_161
        buf323 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_86, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf322, buf323, 1572864, grid=grid(1572864), stream=stream0)
        buf324 = reinterpret_tensor(buf315, (512, 768), (768, 1), 0); del buf315  # reuse
        # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_163, buf323, reinterpret_tensor(primals_162, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf324)
        del primals_163
        # Source Nodes: [hidden_states_87], Original ATen: [aten.native_dropout]
        buf325 = aten.native_dropout(reinterpret_tensor(buf324, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf326 = buf325[0]
        buf327 = buf325[1]
        del buf325
        buf331 = reinterpret_tensor(buf324, (1, 512, 768), (393216, 768, 1), 0); del buf324  # reuse
        buf332 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf417 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32, attention_output_18, hidden_states_89, mixed_query_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf326, buf320, primals_158, primals_159, primals_164, primals_165, buf331, buf332, buf417, 512, 768, grid=grid(512), stream=stream0)
        del primals_159
        buf333 = reinterpret_tensor(buf326, (512, 768), (768, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf332, reinterpret_tensor(primals_166, (768, 768), (1, 768), 0), out=buf333)
        buf334 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf332, reinterpret_tensor(primals_168, (768, 768), (1, 768), 0), out=buf334)
        buf335 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf332, reinterpret_tensor(primals_170, (768, 768), (1, 768), 0), out=buf335)
        buf336 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf333, primals_167, buf336, 393216, grid=grid(393216), stream=stream0)
        del primals_167
        buf337 = reinterpret_tensor(buf333, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf334, primals_169, buf337, 393216, grid=grid(393216), stream=stream0)
        del primals_169
        buf338 = reinterpret_tensor(buf334, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf335, primals_171, buf338, 393216, grid=grid(393216), stream=stream0)
        del primals_171
        # Source Nodes: [], Original ATen: []
        buf339 = aten._scaled_dot_product_efficient_attention(buf336, buf337, buf338, None, True, 0.1, scale=0.125)
        buf340 = buf339[0]
        buf341 = buf339[1]
        buf342 = buf339[2]
        buf343 = buf339[3]
        del buf339
        buf344 = buf335; del buf335  # reuse
        # Source Nodes: [hidden_states_90], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf340, buf344, 393216, grid=grid(393216), stream=stream0)
        buf345 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_173, buf344, reinterpret_tensor(primals_172, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf345)
        del primals_173
        # Source Nodes: [hidden_states_91], Original ATen: [aten.native_dropout]
        buf346 = aten.native_dropout(reinterpret_tensor(buf345, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf347 = buf346[0]
        buf348 = buf346[1]
        del buf346
        buf352 = reinterpret_tensor(buf345, (1, 512, 768), (393216, 768, 1), 0); del buf345  # reuse
        buf353 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf416 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_34, attention_output_20, hidden_states_89, hidden_states_93], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf347, buf331, primals_164, primals_165, primals_174, primals_175, buf352, buf353, buf416, 512, 768, grid=grid(512), stream=stream0)
        del primals_165
        buf354 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf353, reinterpret_tensor(primals_176, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf354)
        del primals_177
        buf355 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf354, buf355, 1572864, grid=grid(1572864), stream=stream0)
        buf356 = reinterpret_tensor(buf347, (512, 768), (768, 1), 0); del buf347  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_179, buf355, reinterpret_tensor(primals_178, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf356)
        del primals_179
        # Source Nodes: [hidden_states_96], Original ATen: [aten.native_dropout]
        buf357 = aten.native_dropout(reinterpret_tensor(buf356, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf358 = buf357[0]
        buf359 = buf357[1]
        del buf357
        buf363 = reinterpret_tensor(buf356, (1, 512, 768), (393216, 768, 1), 0); del buf356  # reuse
        buf364 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf415 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35, attention_output_20, hidden_states_98, mixed_query_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf358, buf352, primals_174, primals_175, primals_180, primals_181, buf363, buf364, buf415, 512, 768, grid=grid(512), stream=stream0)
        del primals_175
        buf365 = reinterpret_tensor(buf358, (512, 768), (768, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf364, reinterpret_tensor(primals_182, (768, 768), (1, 768), 0), out=buf365)
        buf366 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf364, reinterpret_tensor(primals_184, (768, 768), (1, 768), 0), out=buf366)
        buf367 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf364, reinterpret_tensor(primals_186, (768, 768), (1, 768), 0), out=buf367)
        buf368 = empty((1, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf365, primals_183, buf368, 393216, grid=grid(393216), stream=stream0)
        del primals_183
        buf369 = reinterpret_tensor(buf365, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf366, primals_185, buf369, 393216, grid=grid(393216), stream=stream0)
        del primals_185
        buf370 = reinterpret_tensor(buf366, (1, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf367, primals_187, buf370, 393216, grid=grid(393216), stream=stream0)
        del primals_187
        # Source Nodes: [], Original ATen: []
        buf371 = aten._scaled_dot_product_efficient_attention(buf368, buf369, buf370, None, True, 0.1, scale=0.125)
        buf372 = buf371[0]
        buf373 = buf371[1]
        buf374 = buf371[2]
        buf375 = buf371[3]
        del buf371
        buf376 = buf367; del buf367  # reuse
        # Source Nodes: [hidden_states_99], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf372, buf376, 393216, grid=grid(393216), stream=stream0)
        buf377 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_189, buf376, reinterpret_tensor(primals_188, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf377)
        del primals_189
        # Source Nodes: [hidden_states_100], Original ATen: [aten.native_dropout]
        buf378 = aten.native_dropout(reinterpret_tensor(buf377, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf379 = buf378[0]
        buf380 = buf378[1]
        del buf378
        buf384 = reinterpret_tensor(buf377, (1, 512, 768), (393216, 768, 1), 0); del buf377  # reuse
        buf385 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf414 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_37, attention_output_22, hidden_states_102, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf379, buf363, primals_180, primals_181, primals_190, primals_191, buf384, buf385, buf414, 512, 768, grid=grid(512), stream=stream0)
        del primals_181
        buf386 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_193, buf385, reinterpret_tensor(primals_192, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf386)
        del primals_193
        buf387 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_104, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_5.run(buf386, buf387, 1572864, grid=grid(1572864), stream=stream0)
        buf388 = reinterpret_tensor(buf379, (512, 768), (768, 1), 0); del buf379  # reuse
        # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_195, buf387, reinterpret_tensor(primals_194, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf388)
        del primals_195
        # Source Nodes: [hidden_states_105], Original ATen: [aten.native_dropout]
        buf389 = aten.native_dropout(reinterpret_tensor(buf388, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf390 = buf389[0]
        buf391 = buf389[1]
        del buf389
        buf395 = reinterpret_tensor(buf388, (1, 512, 768), (393216, 768, 1), 0); del buf388  # reuse
        buf396 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf413 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, attention_output_22, sequence_output, x_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf390, buf384, primals_190, primals_191, primals_196, primals_197, buf395, buf396, buf413, 512, 768, grid=grid(512), stream=stream0)
        del primals_191
        del primals_197
        buf397 = reinterpret_tensor(buf390, (512, 768), (768, 1), 0); del buf390  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_199, buf396, reinterpret_tensor(primals_198, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf397)
        del primals_199
        buf401 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf402 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf412 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores, x_37, x_38], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7.run(buf397, primals_200, primals_201, buf401, buf402, buf412, 512, 768, grid=grid(512), stream=stream0)
        del primals_201
        buf403 = empty((512, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_203, buf402, reinterpret_tensor(primals_202, (768, 50265), (1, 768), 0), alpha=1, beta=1, out=buf403)
        del primals_203
        buf406 = empty((511, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_8.run(buf403, buf406, 511, 50265, grid=grid(511), stream=stream0)
        buf409 = empty((), device='cuda', dtype=torch.float32)
        buf410 = empty((511, 1), device='cuda', dtype=torch.bool)
        buf411 = empty((511, 1), device='cuda', dtype=torch.int64)
        buf408 = empty((), device='cuda', dtype=torch.float32)
        buf438 = buf409; del buf409  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused_nll_loss_backward_nll_loss_forward_9.run(buf438, primals_205, buf406, buf410, buf411, buf408, 1, 511, grid=grid(1), stream=stream0)
        del primals_205
        return (buf438, reinterpret_tensor(buf403, (1, 512, 50265), (25735680, 50265, 1), 0), primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, primals_204, buf3, buf8, buf12, reinterpret_tensor(buf11, (512, 768), (768, 1), 0), buf16, buf17, buf18, buf21, buf22, buf23, buf20, buf24, buf28, buf32, buf33, buf34, buf35, buf39, buf43, buf44, buf48, buf49, buf50, buf53, buf54, buf55, buf52, buf56, buf60, buf64, buf65, buf66, buf67, buf71, buf75, buf76, buf80, buf81, buf82, buf85, buf86, buf87, buf84, buf88, buf92, buf96, buf97, buf98, buf99, buf103, buf107, buf108, buf112, buf113, buf114, buf117, buf118, buf119, buf116, buf120, buf124, buf128, buf129, buf130, buf131, buf135, buf139, buf140, buf144, buf145, buf146, buf149, buf150, buf151, buf148, buf152, buf156, buf160, buf161, buf162, buf163, buf167, buf171, buf172, buf176, buf177, buf178, buf181, buf182, buf183, buf180, buf184, buf188, buf192, buf193, buf194, buf195, buf199, buf203, buf204, buf208, buf209, buf210, buf213, buf214, buf215, buf212, buf216, buf220, buf224, buf225, buf226, buf227, buf231, buf235, buf236, buf240, buf241, buf242, buf245, buf246, buf247, buf244, buf248, buf252, buf256, buf257, buf258, buf259, buf263, buf267, buf268, buf272, buf273, buf274, buf277, buf278, buf279, buf276, buf280, buf284, buf288, buf289, buf290, buf291, buf295, buf299, buf300, buf304, buf305, buf306, buf309, buf310, buf311, buf308, buf312, buf316, buf320, buf321, buf322, buf323, buf327, buf331, buf332, buf336, buf337, buf338, buf341, buf342, buf343, buf340, buf344, buf348, buf352, buf353, buf354, buf355, buf359, buf363, buf364, buf368, buf369, buf370, buf373, buf374, buf375, buf372, buf376, buf380, buf384, buf385, buf386, buf387, buf391, buf395, buf396, buf397, buf401, buf402, buf406, buf408, buf410, buf411, reinterpret_tensor(primals_202, (50265, 768), (768, 1), 0), buf412, reinterpret_tensor(primals_198, (768, 768), (768, 1), 0), buf413, reinterpret_tensor(primals_194, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_192, (3072, 768), (768, 1), 0), buf414, reinterpret_tensor(primals_188, (768, 768), (768, 1), 0), reinterpret_tensor(primals_186, (768, 768), (768, 1), 0), reinterpret_tensor(primals_184, (768, 768), (768, 1), 0), reinterpret_tensor(primals_182, (768, 768), (768, 1), 0), buf415, reinterpret_tensor(primals_178, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_176, (3072, 768), (768, 1), 0), buf416, reinterpret_tensor(primals_172, (768, 768), (768, 1), 0), reinterpret_tensor(primals_170, (768, 768), (768, 1), 0), reinterpret_tensor(primals_168, (768, 768), (768, 1), 0), reinterpret_tensor(primals_166, (768, 768), (768, 1), 0), buf417, reinterpret_tensor(primals_162, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_160, (3072, 768), (768, 1), 0), buf418, reinterpret_tensor(primals_156, (768, 768), (768, 1), 0), reinterpret_tensor(primals_154, (768, 768), (768, 1), 0), reinterpret_tensor(primals_152, (768, 768), (768, 1), 0), reinterpret_tensor(primals_150, (768, 768), (768, 1), 0), buf419, reinterpret_tensor(primals_146, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_144, (3072, 768), (768, 1), 0), buf420, reinterpret_tensor(primals_140, (768, 768), (768, 1), 0), reinterpret_tensor(primals_138, (768, 768), (768, 1), 0), reinterpret_tensor(primals_136, (768, 768), (768, 1), 0), reinterpret_tensor(primals_134, (768, 768), (768, 1), 0), buf421, reinterpret_tensor(primals_130, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_128, (3072, 768), (768, 1), 0), buf422, reinterpret_tensor(primals_124, (768, 768), (768, 1), 0), reinterpret_tensor(primals_122, (768, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 768), (768, 1), 0), reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), buf423, reinterpret_tensor(primals_114, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_112, (3072, 768), (768, 1), 0), buf424, reinterpret_tensor(primals_108, (768, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), reinterpret_tensor(primals_104, (768, 768), (768, 1), 0), reinterpret_tensor(primals_102, (768, 768), (768, 1), 0), buf425, reinterpret_tensor(primals_98, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_96, (3072, 768), (768, 1), 0), buf426, reinterpret_tensor(primals_92, (768, 768), (768, 1), 0), reinterpret_tensor(primals_90, (768, 768), (768, 1), 0), reinterpret_tensor(primals_88, (768, 768), (768, 1), 0), reinterpret_tensor(primals_86, (768, 768), (768, 1), 0), buf427, reinterpret_tensor(primals_82, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 768), (768, 1), 0), buf428, reinterpret_tensor(primals_76, (768, 768), (768, 1), 0), reinterpret_tensor(primals_74, (768, 768), (768, 1), 0), reinterpret_tensor(primals_72, (768, 768), (768, 1), 0), reinterpret_tensor(primals_70, (768, 768), (768, 1), 0), buf429, reinterpret_tensor(primals_66, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_64, (3072, 768), (768, 1), 0), buf430, reinterpret_tensor(primals_60, (768, 768), (768, 1), 0), reinterpret_tensor(primals_58, (768, 768), (768, 1), 0), reinterpret_tensor(primals_56, (768, 768), (768, 1), 0), reinterpret_tensor(primals_54, (768, 768), (768, 1), 0), buf431, reinterpret_tensor(primals_50, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_48, (3072, 768), (768, 1), 0), buf432, reinterpret_tensor(primals_44, (768, 768), (768, 1), 0), reinterpret_tensor(primals_42, (768, 768), (768, 1), 0), reinterpret_tensor(primals_40, (768, 768), (768, 1), 0), reinterpret_tensor(primals_38, (768, 768), (768, 1), 0), buf433, reinterpret_tensor(primals_34, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_32, (3072, 768), (768, 1), 0), buf434, reinterpret_tensor(primals_28, (768, 768), (768, 1), 0), reinterpret_tensor(primals_26, (768, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 768), (768, 1), 0), buf435, reinterpret_tensor(primals_18, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_16, (3072, 768), (768, 1), 0), buf436, reinterpret_tensor(primals_12, (768, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (768, 1), 0), reinterpret_tensor(primals_6, (768, 768), (768, 1), 0), buf437, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
    primals_198 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((50265, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('RobertaForCausalLM', benchmark_compiled_module)
