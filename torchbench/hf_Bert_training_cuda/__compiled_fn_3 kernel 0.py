
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


# kernel path: /tmp/torchinductor_youkaichao/nd/cndd5ke33ekwz3znvvooekyy4e4gjs4osbotiiwthbggmc2e2ny5.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, mixed_query_layer, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# inputs_embeds => embedding
# mixed_query_layer => view
# position_embeddings => embedding_2
# token_type_embeddings => embedding_1
triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    tmp44 = tmp38 / tmp34
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp16, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp39, rmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp43, rmask)
    tl.store(out_ptr5 + (x3), tmp44, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/gp/cgpiaufgkinpuptgdhgul4xqasnzu22kgwqd2fjbgylhxplipwyt.py
# Source Nodes: [attention_scores], Original ATen: [aten.clone]
# attention_scores => clone_1
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rq/crqakscjemctupahytnoalzfnpqdaqebkcdj44u74gfz5vra7wwo.py
# Source Nodes: [attention_scores], Original ATen: [aten.clone]
# attention_scores => clone_2
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (393216*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfn4kcjg5y2wjvfs4nu7buhwju6qqbrqnridbsfjf2ajfykcsoia.py
# Source Nodes: [attention_probs, attention_probs_1, attention_scores_2], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
# attention_probs => amax, div_1, exp, sub_2, sum_1
# attention_probs_1 => clone_3
# attention_scores_2 => div
triton_per_fused__softmax_add_clone_detach_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_clone_detach_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 24576
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp5, 0))
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp13, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxjxwvh7arozqu32wq3af6nxmv6wqgisa7xvpvldjl65dthtrsm.py
# Source Nodes: [hidden_states], Original ATen: [aten.view]
# hidden_states => view_16
triton_poi_fused_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 512)) + (32768*(x0 // 64)) + (393216*(x1 // 512)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zy/czyzpjh3gdhl5qy7eefnqxevbmsizvwclvcx47wn3ulwbyxqrnc5.py
# Source Nodes: [add_2, attention_output, embeddings_2, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_2 => add_5
# attention_output => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
# embeddings_2 => add_3, mul_2
# hidden_states_3 => view_18
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
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
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp36, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yjatvpkzda4nulwejff3k4md5fdudieuxipkm4765ww5vlbbsq.py
# Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
# hidden_states_5 => view_20
# intermediate_output => add_8, erf, mul_5, mul_6, mul_7
triton_poi_fused_gelu_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
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


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qff6imofgwlf35ngil64473tnv6u3glq4hi3uwtzz6uflucbvi.py
# Source Nodes: [hidden_states_109, hidden_states_111, prediction_scores], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_109 => add_100, erf_12, mul_87, mul_88, mul_89
# hidden_states_111 => add_101, add_102, mul_90, mul_91, rsqrt_25, sub_38, var_mean_25
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp36, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206 = args
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
    assert_size_stride(primals_198, (768, 768), (768, 1))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (30522, 768), (768, 1))
    assert_size_stride(primals_203, (30522, ), (1, ))
    assert_size_stride(primals_204, (1, 512), (512, 1))
    assert_size_stride(primals_205, (1, 512), (512, 1))
    assert_size_stride(primals_206, (4, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf4 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf386 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, inputs_embeds, mixed_query_layer, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0.run(primals_206, primals_1, primals_204, primals_2, primals_205, primals_3, primals_4, primals_5, buf0, buf4, buf5, buf386, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        buf6 = reinterpret_tensor(buf0, (2048, 768), (768, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf5, reinterpret_tensor(primals_6, (768, 768), (1, 768), 0), out=buf6)
        buf7 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf5, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf7)
        buf8 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf5, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf8)
        buf9 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf6, primals_7, buf9, 1572864, grid=grid(1572864), stream=stream0)
        del primals_7
        buf10 = reinterpret_tensor(buf6, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf6  # reuse
        # Source Nodes: [attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf7, primals_9, buf10, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_9
        buf11 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf10, (48, 64, 512), (32768, 512, 1), 0), out=buf11)
        buf14 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf385 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_1, attention_scores_2], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf11, buf14, buf385, 24576, 512, grid=grid(24576), stream=stream0)
        buf15 = reinterpret_tensor(buf7, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf7  # reuse
        # Source Nodes: [context_layer], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf8, primals_11, buf15, 1572864, grid=grid(1572864), stream=stream0)
        del primals_11
        buf16 = reinterpret_tensor(buf8, (48, 512, 64), (32768, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [context_layer], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf14, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf15, (48, 512, 64), (32768, 64, 1), 0), out=buf16)
        buf17 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf16, buf17, 1572864, grid=grid(1572864), stream=stream0)
        buf18 = reinterpret_tensor(buf16, (2048, 768), (768, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf17, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf18)
        buf19 = reinterpret_tensor(buf18, (4, 512, 768), (393216, 768, 1), 0); del buf18  # reuse
        buf23 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf24 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf384 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, attention_output, embeddings_2, hidden_states_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf19, primals_13, buf4, primals_4, primals_5, primals_14, primals_15, buf23, buf24, buf384, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_13
        del primals_5
        buf25 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, buf24, reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del primals_17
        buf26 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_5, intermediate_output], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf25, buf26, 6291456, grid=grid(6291456), stream=stream0)
        buf27 = reinterpret_tensor(buf19, (2048, 768), (768, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf26, reinterpret_tensor(primals_18, (3072, 768), (1, 3072), 0), out=buf27)
        buf28 = reinterpret_tensor(buf27, (4, 512, 768), (393216, 768, 1), 0); del buf27  # reuse
        buf32 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf33 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf383 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_3, attention_output, hidden_states_8, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf28, primals_19, buf23, primals_14, primals_15, primals_20, primals_21, buf32, buf33, buf383, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_15
        del primals_19
        buf34 = reinterpret_tensor(buf28, (2048, 768), (768, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, reinterpret_tensor(primals_22, (768, 768), (1, 768), 0), out=buf34)
        buf35 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, reinterpret_tensor(primals_24, (768, 768), (1, 768), 0), out=buf35)
        buf36 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, reinterpret_tensor(primals_26, (768, 768), (1, 768), 0), out=buf36)
        buf37 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf34, primals_23, buf37, 1572864, grid=grid(1572864), stream=stream0)
        del primals_23
        buf38 = reinterpret_tensor(buf34, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf34  # reuse
        # Source Nodes: [attention_scores_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf35, primals_25, buf38, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_25
        buf39 = buf11; del buf11  # reuse
        # Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf38, (48, 64, 512), (32768, 512, 1), 0), out=buf39)
        buf42 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf382 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_2, attention_probs_3, attention_scores_5], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf39, buf42, buf382, 24576, 512, grid=grid(24576), stream=stream0)
        buf43 = reinterpret_tensor(buf35, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [context_layer_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf36, primals_27, buf43, 1572864, grid=grid(1572864), stream=stream0)
        del primals_27
        buf44 = reinterpret_tensor(buf36, (48, 512, 64), (32768, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf43, (48, 512, 64), (32768, 64, 1), 0), out=buf44)
        buf45 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_9], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf44, buf45, 1572864, grid=grid(1572864), stream=stream0)
        buf46 = reinterpret_tensor(buf44, (2048, 768), (768, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf45, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), out=buf46)
        buf47 = reinterpret_tensor(buf46, (4, 512, 768), (393216, 768, 1), 0); del buf46  # reuse
        buf51 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf52 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf381 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, attention_output_2, hidden_states_12, hidden_states_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf47, primals_29, buf32, primals_20, primals_21, primals_30, primals_31, buf51, buf52, buf381, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_21
        del primals_29
        buf53 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf52, reinterpret_tensor(primals_32, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf53)
        del primals_33
        buf54 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14, intermediate_output_1], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf53, buf54, 6291456, grid=grid(6291456), stream=stream0)
        buf55 = reinterpret_tensor(buf47, (2048, 768), (768, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf54, reinterpret_tensor(primals_34, (3072, 768), (1, 3072), 0), out=buf55)
        buf56 = reinterpret_tensor(buf55, (4, 512, 768), (393216, 768, 1), 0); del buf55  # reuse
        buf60 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf61 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf380 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, attention_output_2, hidden_states_17, mixed_query_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf56, primals_35, buf51, primals_30, primals_31, primals_36, primals_37, buf60, buf61, buf380, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_31
        del primals_35
        buf62 = reinterpret_tensor(buf56, (2048, 768), (768, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf61, reinterpret_tensor(primals_38, (768, 768), (1, 768), 0), out=buf62)
        buf63 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf61, reinterpret_tensor(primals_40, (768, 768), (1, 768), 0), out=buf63)
        buf64 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf61, reinterpret_tensor(primals_42, (768, 768), (1, 768), 0), out=buf64)
        buf65 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf62, primals_39, buf65, 1572864, grid=grid(1572864), stream=stream0)
        del primals_39
        buf66 = reinterpret_tensor(buf62, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf62  # reuse
        # Source Nodes: [attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf63, primals_41, buf66, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_41
        buf67 = buf39; del buf39  # reuse
        # Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf66, (48, 64, 512), (32768, 512, 1), 0), out=buf67)
        buf70 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf379 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_4, attention_probs_5, attention_scores_8], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf67, buf70, buf379, 24576, 512, grid=grid(24576), stream=stream0)
        buf71 = reinterpret_tensor(buf63, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf63  # reuse
        # Source Nodes: [context_layer_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf64, primals_43, buf71, 1572864, grid=grid(1572864), stream=stream0)
        del primals_43
        buf72 = reinterpret_tensor(buf64, (48, 512, 64), (32768, 64, 1), 0); del buf64  # reuse
        # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf71, (48, 512, 64), (32768, 64, 1), 0), out=buf72)
        buf73 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf72, buf73, 1572864, grid=grid(1572864), stream=stream0)
        buf74 = reinterpret_tensor(buf72, (2048, 768), (768, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf73, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), out=buf74)
        buf75 = reinterpret_tensor(buf74, (4, 512, 768), (393216, 768, 1), 0); del buf74  # reuse
        buf79 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf80 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf378 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, attention_output_4, hidden_states_17, hidden_states_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf75, primals_45, buf60, primals_36, primals_37, primals_46, primals_47, buf79, buf80, buf378, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_37
        del primals_45
        buf81 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, buf80, reinterpret_tensor(primals_48, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf81)
        del primals_49
        buf82 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_23, intermediate_output_2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf81, buf82, 6291456, grid=grid(6291456), stream=stream0)
        buf83 = reinterpret_tensor(buf75, (2048, 768), (768, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf82, reinterpret_tensor(primals_50, (3072, 768), (1, 3072), 0), out=buf83)
        buf84 = reinterpret_tensor(buf83, (4, 512, 768), (393216, 768, 1), 0); del buf83  # reuse
        buf88 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf89 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf377 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, attention_output_4, hidden_states_26, mixed_query_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf84, primals_51, buf79, primals_46, primals_47, primals_52, primals_53, buf88, buf89, buf377, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_47
        del primals_51
        buf90 = reinterpret_tensor(buf84, (2048, 768), (768, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf89, reinterpret_tensor(primals_54, (768, 768), (1, 768), 0), out=buf90)
        buf91 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf89, reinterpret_tensor(primals_56, (768, 768), (1, 768), 0), out=buf91)
        buf92 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf89, reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), out=buf92)
        buf93 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf90, primals_55, buf93, 1572864, grid=grid(1572864), stream=stream0)
        del primals_55
        buf94 = reinterpret_tensor(buf90, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf90  # reuse
        # Source Nodes: [attention_scores_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf91, primals_57, buf94, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_57
        buf95 = buf67; del buf67  # reuse
        # Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf94, (48, 64, 512), (32768, 512, 1), 0), out=buf95)
        buf98 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf376 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_6, attention_probs_7, attention_scores_11], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf95, buf98, buf376, 24576, 512, grid=grid(24576), stream=stream0)
        buf99 = reinterpret_tensor(buf91, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf91  # reuse
        # Source Nodes: [context_layer_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf92, primals_59, buf99, 1572864, grid=grid(1572864), stream=stream0)
        del primals_59
        buf100 = reinterpret_tensor(buf92, (48, 512, 64), (32768, 64, 1), 0); del buf92  # reuse
        # Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf99, (48, 512, 64), (32768, 64, 1), 0), out=buf100)
        buf101 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_27], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf100, buf101, 1572864, grid=grid(1572864), stream=stream0)
        buf102 = reinterpret_tensor(buf100, (2048, 768), (768, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf101, reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), out=buf102)
        buf103 = reinterpret_tensor(buf102, (4, 512, 768), (393216, 768, 1), 0); del buf102  # reuse
        buf107 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf108 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf375 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, attention_output_6, hidden_states_26, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf103, primals_61, buf88, primals_52, primals_53, primals_62, primals_63, buf107, buf108, buf375, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_53
        del primals_61
        buf109 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf108, reinterpret_tensor(primals_64, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf109)
        del primals_65
        buf110 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, intermediate_output_3], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf109, buf110, 6291456, grid=grid(6291456), stream=stream0)
        buf111 = reinterpret_tensor(buf103, (2048, 768), (768, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf110, reinterpret_tensor(primals_66, (3072, 768), (1, 3072), 0), out=buf111)
        buf112 = reinterpret_tensor(buf111, (4, 512, 768), (393216, 768, 1), 0); del buf111  # reuse
        buf116 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf117 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf374 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, attention_output_6, hidden_states_35, mixed_query_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf112, primals_67, buf107, primals_62, primals_63, primals_68, primals_69, buf116, buf117, buf374, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_63
        del primals_67
        buf118 = reinterpret_tensor(buf112, (2048, 768), (768, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf117, reinterpret_tensor(primals_70, (768, 768), (1, 768), 0), out=buf118)
        buf119 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf117, reinterpret_tensor(primals_72, (768, 768), (1, 768), 0), out=buf119)
        buf120 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf117, reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), out=buf120)
        buf121 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf118, primals_71, buf121, 1572864, grid=grid(1572864), stream=stream0)
        del primals_71
        buf122 = reinterpret_tensor(buf118, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf118  # reuse
        # Source Nodes: [attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf119, primals_73, buf122, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_73
        buf123 = buf95; del buf95  # reuse
        # Source Nodes: [attention_scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf122, (48, 64, 512), (32768, 512, 1), 0), out=buf123)
        buf126 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf373 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_8, attention_probs_9, attention_scores_14], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf123, buf126, buf373, 24576, 512, grid=grid(24576), stream=stream0)
        buf127 = reinterpret_tensor(buf119, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf119  # reuse
        # Source Nodes: [context_layer_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf120, primals_75, buf127, 1572864, grid=grid(1572864), stream=stream0)
        del primals_75
        buf128 = reinterpret_tensor(buf120, (48, 512, 64), (32768, 64, 1), 0); del buf120  # reuse
        # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf126, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf127, (48, 512, 64), (32768, 64, 1), 0), out=buf128)
        buf129 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf128, buf129, 1572864, grid=grid(1572864), stream=stream0)
        buf130 = reinterpret_tensor(buf128, (2048, 768), (768, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf129, reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), out=buf130)
        buf131 = reinterpret_tensor(buf130, (4, 512, 768), (393216, 768, 1), 0); del buf130  # reuse
        buf135 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf136 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf372 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, attention_output_8, hidden_states_35, hidden_states_39], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf131, primals_77, buf116, primals_68, primals_69, primals_78, primals_79, buf135, buf136, buf372, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_69
        del primals_77
        buf137 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf136, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf137)
        del primals_81
        buf138 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_41, intermediate_output_4], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf137, buf138, 6291456, grid=grid(6291456), stream=stream0)
        buf139 = reinterpret_tensor(buf131, (2048, 768), (768, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf138, reinterpret_tensor(primals_82, (3072, 768), (1, 3072), 0), out=buf139)
        buf140 = reinterpret_tensor(buf139, (4, 512, 768), (393216, 768, 1), 0); del buf139  # reuse
        buf144 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf145 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf371 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, attention_output_8, hidden_states_44, mixed_query_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf140, primals_83, buf135, primals_78, primals_79, primals_84, primals_85, buf144, buf145, buf371, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_79
        del primals_83
        buf146 = reinterpret_tensor(buf140, (2048, 768), (768, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf145, reinterpret_tensor(primals_86, (768, 768), (1, 768), 0), out=buf146)
        buf147 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf145, reinterpret_tensor(primals_88, (768, 768), (1, 768), 0), out=buf147)
        buf148 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf145, reinterpret_tensor(primals_90, (768, 768), (1, 768), 0), out=buf148)
        buf149 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf146, primals_87, buf149, 1572864, grid=grid(1572864), stream=stream0)
        del primals_87
        buf150 = reinterpret_tensor(buf146, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf146  # reuse
        # Source Nodes: [attention_scores_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf147, primals_89, buf150, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_89
        buf151 = buf123; del buf123  # reuse
        # Source Nodes: [attention_scores_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf150, (48, 64, 512), (32768, 512, 1), 0), out=buf151)
        buf154 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf370 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_10, attention_probs_11, attention_scores_17], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf151, buf154, buf370, 24576, 512, grid=grid(24576), stream=stream0)
        buf155 = reinterpret_tensor(buf147, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf147  # reuse
        # Source Nodes: [context_layer_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf148, primals_91, buf155, 1572864, grid=grid(1572864), stream=stream0)
        del primals_91
        buf156 = reinterpret_tensor(buf148, (48, 512, 64), (32768, 64, 1), 0); del buf148  # reuse
        # Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf155, (48, 512, 64), (32768, 64, 1), 0), out=buf156)
        buf157 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_45], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf156, buf157, 1572864, grid=grid(1572864), stream=stream0)
        buf158 = reinterpret_tensor(buf156, (2048, 768), (768, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf157, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), out=buf158)
        buf159 = reinterpret_tensor(buf158, (4, 512, 768), (393216, 768, 1), 0); del buf158  # reuse
        buf163 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf164 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf369 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, attention_output_10, hidden_states_44, hidden_states_48], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf159, primals_93, buf144, primals_84, primals_85, primals_94, primals_95, buf163, buf164, buf369, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_85
        del primals_93
        buf165 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, buf164, reinterpret_tensor(primals_96, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf165)
        del primals_97
        buf166 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50, intermediate_output_5], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf165, buf166, 6291456, grid=grid(6291456), stream=stream0)
        buf167 = reinterpret_tensor(buf159, (2048, 768), (768, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf166, reinterpret_tensor(primals_98, (3072, 768), (1, 3072), 0), out=buf167)
        buf168 = reinterpret_tensor(buf167, (4, 512, 768), (393216, 768, 1), 0); del buf167  # reuse
        buf172 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf173 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf368 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, attention_output_10, hidden_states_53, mixed_query_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf168, primals_99, buf163, primals_94, primals_95, primals_100, primals_101, buf172, buf173, buf368, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_95
        del primals_99
        buf174 = reinterpret_tensor(buf168, (2048, 768), (768, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf173, reinterpret_tensor(primals_102, (768, 768), (1, 768), 0), out=buf174)
        buf175 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf173, reinterpret_tensor(primals_104, (768, 768), (1, 768), 0), out=buf175)
        buf176 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf173, reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), out=buf176)
        buf177 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf174, primals_103, buf177, 1572864, grid=grid(1572864), stream=stream0)
        del primals_103
        buf178 = reinterpret_tensor(buf174, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf174  # reuse
        # Source Nodes: [attention_scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf175, primals_105, buf178, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_105
        buf179 = buf151; del buf151  # reuse
        # Source Nodes: [attention_scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf177, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf178, (48, 64, 512), (32768, 512, 1), 0), out=buf179)
        buf182 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf367 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_12, attention_probs_13, attention_scores_20], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf179, buf182, buf367, 24576, 512, grid=grid(24576), stream=stream0)
        buf183 = reinterpret_tensor(buf175, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf175  # reuse
        # Source Nodes: [context_layer_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf176, primals_107, buf183, 1572864, grid=grid(1572864), stream=stream0)
        del primals_107
        buf184 = reinterpret_tensor(buf176, (48, 512, 64), (32768, 64, 1), 0); del buf176  # reuse
        # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf183, (48, 512, 64), (32768, 64, 1), 0), out=buf184)
        buf185 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_54], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf184, buf185, 1572864, grid=grid(1572864), stream=stream0)
        buf186 = reinterpret_tensor(buf184, (2048, 768), (768, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf185, reinterpret_tensor(primals_108, (768, 768), (1, 768), 0), out=buf186)
        buf187 = reinterpret_tensor(buf186, (4, 512, 768), (393216, 768, 1), 0); del buf186  # reuse
        buf191 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf192 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf366 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, attention_output_12, hidden_states_53, hidden_states_57], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf187, primals_109, buf172, primals_100, primals_101, primals_110, primals_111, buf191, buf192, buf366, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_101
        del primals_109
        buf193 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_113, buf192, reinterpret_tensor(primals_112, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf193)
        del primals_113
        buf194 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_59, intermediate_output_6], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf193, buf194, 6291456, grid=grid(6291456), stream=stream0)
        buf195 = reinterpret_tensor(buf187, (2048, 768), (768, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf194, reinterpret_tensor(primals_114, (3072, 768), (1, 3072), 0), out=buf195)
        buf196 = reinterpret_tensor(buf195, (4, 512, 768), (393216, 768, 1), 0); del buf195  # reuse
        buf200 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf201 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf365 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, attention_output_12, hidden_states_62, mixed_query_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf196, primals_115, buf191, primals_110, primals_111, primals_116, primals_117, buf200, buf201, buf365, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_111
        del primals_115
        buf202 = reinterpret_tensor(buf196, (2048, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), out=buf202)
        buf203 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_120, (768, 768), (1, 768), 0), out=buf203)
        buf204 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_122, (768, 768), (1, 768), 0), out=buf204)
        buf205 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf202, primals_119, buf205, 1572864, grid=grid(1572864), stream=stream0)
        del primals_119
        buf206 = reinterpret_tensor(buf202, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf202  # reuse
        # Source Nodes: [attention_scores_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf203, primals_121, buf206, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_121
        buf207 = buf179; del buf179  # reuse
        # Source Nodes: [attention_scores_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf205, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf206, (48, 64, 512), (32768, 512, 1), 0), out=buf207)
        buf210 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf364 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_14, attention_probs_15, attention_scores_23], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf207, buf210, buf364, 24576, 512, grid=grid(24576), stream=stream0)
        buf211 = reinterpret_tensor(buf203, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [context_layer_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf204, primals_123, buf211, 1572864, grid=grid(1572864), stream=stream0)
        del primals_123
        buf212 = reinterpret_tensor(buf204, (48, 512, 64), (32768, 64, 1), 0); del buf204  # reuse
        # Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf210, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf211, (48, 512, 64), (32768, 64, 1), 0), out=buf212)
        buf213 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_63], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf212, buf213, 1572864, grid=grid(1572864), stream=stream0)
        buf214 = reinterpret_tensor(buf212, (2048, 768), (768, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf213, reinterpret_tensor(primals_124, (768, 768), (1, 768), 0), out=buf214)
        buf215 = reinterpret_tensor(buf214, (4, 512, 768), (393216, 768, 1), 0); del buf214  # reuse
        buf219 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf220 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf363 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, attention_output_14, hidden_states_62, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf215, primals_125, buf200, primals_116, primals_117, primals_126, primals_127, buf219, buf220, buf363, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_117
        del primals_125
        buf221 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_129, buf220, reinterpret_tensor(primals_128, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf221)
        del primals_129
        buf222 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_68, intermediate_output_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf221, buf222, 6291456, grid=grid(6291456), stream=stream0)
        buf223 = reinterpret_tensor(buf215, (2048, 768), (768, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf222, reinterpret_tensor(primals_130, (3072, 768), (1, 3072), 0), out=buf223)
        buf224 = reinterpret_tensor(buf223, (4, 512, 768), (393216, 768, 1), 0); del buf223  # reuse
        buf228 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf229 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf362 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_24, attention_output_14, hidden_states_71, mixed_query_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf224, primals_131, buf219, primals_126, primals_127, primals_132, primals_133, buf228, buf229, buf362, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_127
        del primals_131
        buf230 = reinterpret_tensor(buf224, (2048, 768), (768, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf229, reinterpret_tensor(primals_134, (768, 768), (1, 768), 0), out=buf230)
        buf231 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf229, reinterpret_tensor(primals_136, (768, 768), (1, 768), 0), out=buf231)
        buf232 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf229, reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), out=buf232)
        buf233 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf230, primals_135, buf233, 1572864, grid=grid(1572864), stream=stream0)
        del primals_135
        buf234 = reinterpret_tensor(buf230, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf230  # reuse
        # Source Nodes: [attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf231, primals_137, buf234, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_137
        buf235 = buf207; del buf207  # reuse
        # Source Nodes: [attention_scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf233, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf234, (48, 64, 512), (32768, 512, 1), 0), out=buf235)
        buf238 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf361 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_16, attention_probs_17, attention_scores_26], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf235, buf238, buf361, 24576, 512, grid=grid(24576), stream=stream0)
        buf239 = reinterpret_tensor(buf231, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf231  # reuse
        # Source Nodes: [context_layer_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf232, primals_139, buf239, 1572864, grid=grid(1572864), stream=stream0)
        del primals_139
        buf240 = reinterpret_tensor(buf232, (48, 512, 64), (32768, 64, 1), 0); del buf232  # reuse
        # Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf238, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf239, (48, 512, 64), (32768, 64, 1), 0), out=buf240)
        buf241 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf240, buf241, 1572864, grid=grid(1572864), stream=stream0)
        buf242 = reinterpret_tensor(buf240, (2048, 768), (768, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf241, reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), out=buf242)
        buf243 = reinterpret_tensor(buf242, (4, 512, 768), (393216, 768, 1), 0); del buf242  # reuse
        buf247 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf248 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf360 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, attention_output_16, hidden_states_71, hidden_states_75], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf243, primals_141, buf228, primals_132, primals_133, primals_142, primals_143, buf247, buf248, buf360, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_133
        del primals_141
        buf249 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, buf248, reinterpret_tensor(primals_144, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf249)
        del primals_145
        buf250 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_77, intermediate_output_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf249, buf250, 6291456, grid=grid(6291456), stream=stream0)
        buf251 = reinterpret_tensor(buf243, (2048, 768), (768, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf250, reinterpret_tensor(primals_146, (3072, 768), (1, 3072), 0), out=buf251)
        buf252 = reinterpret_tensor(buf251, (4, 512, 768), (393216, 768, 1), 0); del buf251  # reuse
        buf256 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf257 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf359 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, attention_output_16, hidden_states_80, mixed_query_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf252, primals_147, buf247, primals_142, primals_143, primals_148, primals_149, buf256, buf257, buf359, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_143
        del primals_147
        buf258 = reinterpret_tensor(buf252, (2048, 768), (768, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf257, reinterpret_tensor(primals_150, (768, 768), (1, 768), 0), out=buf258)
        buf259 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf257, reinterpret_tensor(primals_152, (768, 768), (1, 768), 0), out=buf259)
        buf260 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf257, reinterpret_tensor(primals_154, (768, 768), (1, 768), 0), out=buf260)
        buf261 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf258, primals_151, buf261, 1572864, grid=grid(1572864), stream=stream0)
        del primals_151
        buf262 = reinterpret_tensor(buf258, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf258  # reuse
        # Source Nodes: [attention_scores_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf259, primals_153, buf262, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_153
        buf263 = buf235; del buf235  # reuse
        # Source Nodes: [attention_scores_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf262, (48, 64, 512), (32768, 512, 1), 0), out=buf263)
        buf266 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf358 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_18, attention_probs_19, attention_scores_29], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf263, buf266, buf358, 24576, 512, grid=grid(24576), stream=stream0)
        buf267 = reinterpret_tensor(buf259, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [context_layer_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf260, primals_155, buf267, 1572864, grid=grid(1572864), stream=stream0)
        del primals_155
        buf268 = reinterpret_tensor(buf260, (48, 512, 64), (32768, 64, 1), 0); del buf260  # reuse
        # Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf266, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf267, (48, 512, 64), (32768, 64, 1), 0), out=buf268)
        buf269 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_81], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf268, buf269, 1572864, grid=grid(1572864), stream=stream0)
        buf270 = reinterpret_tensor(buf268, (2048, 768), (768, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf269, reinterpret_tensor(primals_156, (768, 768), (1, 768), 0), out=buf270)
        buf271 = reinterpret_tensor(buf270, (4, 512, 768), (393216, 768, 1), 0); del buf270  # reuse
        buf275 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf276 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf357 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, attention_output_18, hidden_states_80, hidden_states_84], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf271, primals_157, buf256, primals_148, primals_149, primals_158, primals_159, buf275, buf276, buf357, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_149
        del primals_157
        buf277 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf276, reinterpret_tensor(primals_160, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf277)
        del primals_161
        buf278 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_86, intermediate_output_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf277, buf278, 6291456, grid=grid(6291456), stream=stream0)
        buf279 = reinterpret_tensor(buf271, (2048, 768), (768, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf278, reinterpret_tensor(primals_162, (3072, 768), (1, 3072), 0), out=buf279)
        buf280 = reinterpret_tensor(buf279, (4, 512, 768), (393216, 768, 1), 0); del buf279  # reuse
        buf284 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf285 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf356 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, attention_output_18, hidden_states_89, mixed_query_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf280, primals_163, buf275, primals_158, primals_159, primals_164, primals_165, buf284, buf285, buf356, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_159
        del primals_163
        buf286 = reinterpret_tensor(buf280, (2048, 768), (768, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf285, reinterpret_tensor(primals_166, (768, 768), (1, 768), 0), out=buf286)
        buf287 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf285, reinterpret_tensor(primals_168, (768, 768), (1, 768), 0), out=buf287)
        buf288 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf285, reinterpret_tensor(primals_170, (768, 768), (1, 768), 0), out=buf288)
        buf289 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf286, primals_167, buf289, 1572864, grid=grid(1572864), stream=stream0)
        del primals_167
        buf290 = reinterpret_tensor(buf286, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf286  # reuse
        # Source Nodes: [attention_scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf287, primals_169, buf290, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_169
        buf291 = buf263; del buf263  # reuse
        # Source Nodes: [attention_scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf290, (48, 64, 512), (32768, 512, 1), 0), out=buf291)
        buf294 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf355 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_20, attention_probs_21, attention_scores_32], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf291, buf294, buf355, 24576, 512, grid=grid(24576), stream=stream0)
        buf295 = reinterpret_tensor(buf287, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf287  # reuse
        # Source Nodes: [context_layer_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf288, primals_171, buf295, 1572864, grid=grid(1572864), stream=stream0)
        del primals_171
        buf296 = reinterpret_tensor(buf288, (48, 512, 64), (32768, 64, 1), 0); del buf288  # reuse
        # Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf295, (48, 512, 64), (32768, 64, 1), 0), out=buf296)
        buf297 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf296, buf297, 1572864, grid=grid(1572864), stream=stream0)
        buf298 = reinterpret_tensor(buf296, (2048, 768), (768, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_172, (768, 768), (1, 768), 0), out=buf298)
        buf299 = reinterpret_tensor(buf298, (4, 512, 768), (393216, 768, 1), 0); del buf298  # reuse
        buf303 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf304 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf354 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32, attention_output_20, hidden_states_89, hidden_states_93], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf299, primals_173, buf284, primals_164, primals_165, primals_174, primals_175, buf303, buf304, buf354, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_165
        del primals_173
        buf305 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf304, reinterpret_tensor(primals_176, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf305)
        del primals_177
        buf306 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, intermediate_output_10], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf305, buf306, 6291456, grid=grid(6291456), stream=stream0)
        buf307 = reinterpret_tensor(buf299, (2048, 768), (768, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf306, reinterpret_tensor(primals_178, (3072, 768), (1, 3072), 0), out=buf307)
        buf308 = reinterpret_tensor(buf307, (4, 512, 768), (393216, 768, 1), 0); del buf307  # reuse
        buf312 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf313 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf353 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, attention_output_20, hidden_states_98, mixed_query_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf308, primals_179, buf303, primals_174, primals_175, primals_180, primals_181, buf312, buf313, buf353, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_175
        del primals_179
        buf314 = reinterpret_tensor(buf308, (2048, 768), (768, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf313, reinterpret_tensor(primals_182, (768, 768), (1, 768), 0), out=buf314)
        buf315 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf313, reinterpret_tensor(primals_184, (768, 768), (1, 768), 0), out=buf315)
        buf316 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf313, reinterpret_tensor(primals_186, (768, 768), (1, 768), 0), out=buf316)
        buf317 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf314, primals_183, buf317, 1572864, grid=grid(1572864), stream=stream0)
        del primals_183
        buf318 = reinterpret_tensor(buf314, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf314  # reuse
        # Source Nodes: [attention_scores_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf315, primals_185, buf318, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_185
        buf319 = buf291; del buf291  # reuse
        # Source Nodes: [attention_scores_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf317, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf318, (48, 64, 512), (32768, 512, 1), 0), out=buf319)
        buf322 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf352 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_22, attention_probs_23, attention_scores_35], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf319, buf322, buf352, 24576, 512, grid=grid(24576), stream=stream0)
        del buf319
        buf323 = reinterpret_tensor(buf315, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf315  # reuse
        # Source Nodes: [context_layer_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf316, primals_187, buf323, 1572864, grid=grid(1572864), stream=stream0)
        del primals_187
        buf324 = reinterpret_tensor(buf316, (48, 512, 64), (32768, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf322, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf323, (48, 512, 64), (32768, 64, 1), 0), out=buf324)
        buf325 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_99], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf324, buf325, 1572864, grid=grid(1572864), stream=stream0)
        buf326 = reinterpret_tensor(buf324, (2048, 768), (768, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf325, reinterpret_tensor(primals_188, (768, 768), (1, 768), 0), out=buf326)
        buf327 = reinterpret_tensor(buf326, (4, 512, 768), (393216, 768, 1), 0); del buf326  # reuse
        buf331 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf332 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf351 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35, attention_output_22, hidden_states_102, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf327, primals_189, buf312, primals_180, primals_181, primals_190, primals_191, buf331, buf332, buf351, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_181
        del primals_189
        buf333 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_193, buf332, reinterpret_tensor(primals_192, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf333)
        del primals_193
        buf334 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_104, intermediate_output_11], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf333, buf334, 6291456, grid=grid(6291456), stream=stream0)
        buf335 = reinterpret_tensor(buf327, (2048, 768), (768, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf334, reinterpret_tensor(primals_194, (3072, 768), (1, 3072), 0), out=buf335)
        buf336 = reinterpret_tensor(buf335, (4, 512, 768), (393216, 768, 1), 0); del buf335  # reuse
        buf340 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf341 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf350 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, attention_output_22, hidden_states_108, sequence_output], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf336, primals_195, buf331, primals_190, primals_191, primals_196, primals_197, buf340, buf341, buf350, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_191
        del primals_195
        del primals_197
        buf342 = reinterpret_tensor(buf336, (2048, 768), (768, 1), 0); del buf336  # reuse
        # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_199, buf341, reinterpret_tensor(primals_198, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf342)
        del primals_199
        buf346 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf347 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf349 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109, hidden_states_111, prediction_scores], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7.run(buf342, primals_200, primals_201, buf346, buf347, buf349, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_201
        buf348 = empty((2048, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_203, buf347, reinterpret_tensor(primals_202, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf348)
        del primals_203
        return (reinterpret_tensor(buf348, (4, 512, 30522), (15627264, 30522, 1), 0), primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, reinterpret_tensor(primals_204, (4, 512), (0, 1), 0), primals_205, buf4, buf5, buf17, buf23, buf24, buf25, buf26, buf32, buf33, buf45, buf51, buf52, buf53, buf54, buf60, buf61, buf73, buf79, buf80, buf81, buf82, buf88, buf89, buf101, buf107, buf108, buf109, buf110, buf116, buf117, buf129, buf135, buf136, buf137, buf138, buf144, buf145, buf157, buf163, buf164, buf165, buf166, buf172, buf173, buf185, buf191, buf192, buf193, buf194, buf200, buf201, buf213, buf219, buf220, buf221, buf222, buf228, buf229, buf241, buf247, buf248, buf249, buf250, buf256, buf257, buf269, buf275, buf276, buf277, buf278, buf284, buf285, buf297, buf303, buf304, buf305, buf306, buf312, buf313, buf325, buf331, buf332, buf333, buf334, buf340, buf341, buf342, buf346, buf347, reinterpret_tensor(primals_202, (30522, 768), (768, 1), 0), buf349, reinterpret_tensor(primals_198, (768, 768), (768, 1), 0), buf350, reinterpret_tensor(primals_194, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_192, (3072, 768), (768, 1), 0), buf351, reinterpret_tensor(primals_188, (768, 768), (768, 1), 0), reinterpret_tensor(buf322, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf323, (48, 64, 512), (32768, 1, 64), 0), buf352, reinterpret_tensor(buf317, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf318, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_186, (768, 768), (768, 1), 0), reinterpret_tensor(primals_184, (768, 768), (768, 1), 0), reinterpret_tensor(primals_182, (768, 768), (768, 1), 0), buf353, reinterpret_tensor(primals_178, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_176, (3072, 768), (768, 1), 0), buf354, reinterpret_tensor(primals_172, (768, 768), (768, 1), 0), reinterpret_tensor(buf294, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf295, (48, 64, 512), (32768, 1, 64), 0), buf355, reinterpret_tensor(buf289, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf290, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_170, (768, 768), (768, 1), 0), reinterpret_tensor(primals_168, (768, 768), (768, 1), 0), reinterpret_tensor(primals_166, (768, 768), (768, 1), 0), buf356, reinterpret_tensor(primals_162, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_160, (3072, 768), (768, 1), 0), buf357, reinterpret_tensor(primals_156, (768, 768), (768, 1), 0), reinterpret_tensor(buf266, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf267, (48, 64, 512), (32768, 1, 64), 0), buf358, reinterpret_tensor(buf261, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf262, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_154, (768, 768), (768, 1), 0), reinterpret_tensor(primals_152, (768, 768), (768, 1), 0), reinterpret_tensor(primals_150, (768, 768), (768, 1), 0), buf359, reinterpret_tensor(primals_146, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_144, (3072, 768), (768, 1), 0), buf360, reinterpret_tensor(primals_140, (768, 768), (768, 1), 0), reinterpret_tensor(buf238, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf239, (48, 64, 512), (32768, 1, 64), 0), buf361, reinterpret_tensor(buf233, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf234, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_138, (768, 768), (768, 1), 0), reinterpret_tensor(primals_136, (768, 768), (768, 1), 0), reinterpret_tensor(primals_134, (768, 768), (768, 1), 0), buf362, reinterpret_tensor(primals_130, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_128, (3072, 768), (768, 1), 0), buf363, reinterpret_tensor(primals_124, (768, 768), (768, 1), 0), reinterpret_tensor(buf210, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf211, (48, 64, 512), (32768, 1, 64), 0), buf364, reinterpret_tensor(buf205, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf206, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_122, (768, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 768), (768, 1), 0), reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), buf365, reinterpret_tensor(primals_114, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_112, (3072, 768), (768, 1), 0), buf366, reinterpret_tensor(primals_108, (768, 768), (768, 1), 0), reinterpret_tensor(buf182, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf183, (48, 64, 512), (32768, 1, 64), 0), buf367, reinterpret_tensor(buf177, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf178, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), reinterpret_tensor(primals_104, (768, 768), (768, 1), 0), reinterpret_tensor(primals_102, (768, 768), (768, 1), 0), buf368, reinterpret_tensor(primals_98, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_96, (3072, 768), (768, 1), 0), buf369, reinterpret_tensor(primals_92, (768, 768), (768, 1), 0), reinterpret_tensor(buf154, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf155, (48, 64, 512), (32768, 1, 64), 0), buf370, reinterpret_tensor(buf149, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf150, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_90, (768, 768), (768, 1), 0), reinterpret_tensor(primals_88, (768, 768), (768, 1), 0), reinterpret_tensor(primals_86, (768, 768), (768, 1), 0), buf371, reinterpret_tensor(primals_82, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 768), (768, 1), 0), buf372, reinterpret_tensor(primals_76, (768, 768), (768, 1), 0), reinterpret_tensor(buf126, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf127, (48, 64, 512), (32768, 1, 64), 0), buf373, reinterpret_tensor(buf121, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf122, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_74, (768, 768), (768, 1), 0), reinterpret_tensor(primals_72, (768, 768), (768, 1), 0), reinterpret_tensor(primals_70, (768, 768), (768, 1), 0), buf374, reinterpret_tensor(primals_66, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_64, (3072, 768), (768, 1), 0), buf375, reinterpret_tensor(primals_60, (768, 768), (768, 1), 0), reinterpret_tensor(buf98, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf99, (48, 64, 512), (32768, 1, 64), 0), buf376, reinterpret_tensor(buf93, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf94, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_58, (768, 768), (768, 1), 0), reinterpret_tensor(primals_56, (768, 768), (768, 1), 0), reinterpret_tensor(primals_54, (768, 768), (768, 1), 0), buf377, reinterpret_tensor(primals_50, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_48, (3072, 768), (768, 1), 0), buf378, reinterpret_tensor(primals_44, (768, 768), (768, 1), 0), reinterpret_tensor(buf70, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf71, (48, 64, 512), (32768, 1, 64), 0), buf379, reinterpret_tensor(buf65, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf66, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_42, (768, 768), (768, 1), 0), reinterpret_tensor(primals_40, (768, 768), (768, 1), 0), reinterpret_tensor(primals_38, (768, 768), (768, 1), 0), buf380, reinterpret_tensor(primals_34, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_32, (3072, 768), (768, 1), 0), buf381, reinterpret_tensor(primals_28, (768, 768), (768, 1), 0), reinterpret_tensor(buf42, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf43, (48, 64, 512), (32768, 1, 64), 0), buf382, reinterpret_tensor(buf37, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf38, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_26, (768, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 768), (768, 1), 0), buf383, reinterpret_tensor(primals_18, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_16, (3072, 768), (768, 1), 0), buf384, reinterpret_tensor(primals_12, (768, 768), (768, 1), 0), reinterpret_tensor(buf14, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf15, (48, 64, 512), (32768, 1, 64), 0), buf385, reinterpret_tensor(buf9, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf10, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (768, 1), 0), reinterpret_tensor(primals_6, (768, 768), (768, 1), 0), buf386, )


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
    primals_198 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Bert', benchmark_compiled_module)
