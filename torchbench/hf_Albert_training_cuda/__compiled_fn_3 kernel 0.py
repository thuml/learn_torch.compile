
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


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5cobb4iu233zswno746gf257ancktgpsg5kkyid7uegbt22txl.py
# Source Nodes: [embeddings, embeddings_1, embeddings_2, hidden_states, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# embeddings => add
# embeddings_1 => add_1
# embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
# hidden_states => view
# inputs_embeds => embedding
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp44 = tmp38 / tmp34
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp16, rmask)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp39, rmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp43, rmask)
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
# Source Nodes: [projected_context_layer], Original ATen: [aten.view]
# projected_context_layer => view_18
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


# kernel path: /tmp/torchinductor_youkaichao/fh/cfh6uhw3n3k4cwituabb55orglw7ilb2yhkturuu3jllspqcqdyy.py
# Source Nodes: [add_2, ffn_output, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_2 => add_5
# ffn_output => view_20
# layernormed_context_layer => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckbzyepjcvqef6rf57gck2uykbgalg45oc4n3sbcnkmw33vebfzj.py
# Source Nodes: [add_3, add_4, ffn_output_1, ffn_output_3, mul_1, mul_2, mul_3, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
# add_3 => add_8
# add_4 => add_9
# ffn_output_1 => mul_8
# ffn_output_3 => view_22
# mul_1 => mul_5
# mul_2 => mul_6
# mul_3 => mul_7
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_view_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_view_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fj/cfjzmei3wjpuquusdscmbpa34sazbs25sfs7tcode3dbep2zuzbj.py
# Source Nodes: [add_5, hidden_states_3, layernormed_context_layer, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_5 => add_10
# hidden_states_3 => add_11, add_12, mul_10, mul_9, rsqrt_2, sub_4, var_mean_2
# layernormed_context_layer => add_7, mul_4
# mixed_query_layer_1 => view_24
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/iu/ciu3htspr4exri4u52wumdw2hcq7ys32ww7sbk6dk2gxqpndwcez.py
# Source Nodes: [add_7, ffn_output_4, hidden_states_3, layernormed_context_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_7 => add_14
# ffn_output_4 => view_42
# hidden_states_3 => add_12, mul_10
# layernormed_context_layer_1 => add_15, add_16, mul_11, mul_12, rsqrt_3, sub_6, var_mean_3
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
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


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3yvvfw63izrsvgcypoj2ismstc6rto3efkfkruc5dvtb7qfoka.py
# Source Nodes: [add_61, add_62, hidden_states_38, hidden_states_39, mul_49, mul_50, mul_51, pow_13, prediction_scores, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh, aten.view]
# add_61 => add_112
# add_62 => add_113
# hidden_states_38 => mul_102
# hidden_states_39 => add_114, add_115, mul_103, mul_104, rsqrt_25, sub_38, var_mean_25
# mul_49 => mul_99
# mul_50 => mul_100
# mul_51 => mul_101
# pow_13 => pow_13
# prediction_scores => view_268
# tanh_12 => tanh_12
triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp37 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = 128.0
    tmp31 = tmp29 / tmp30
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp13 - tmp23
    tmp36 = tmp35 * tmp34
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp8, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp34, None)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp40, rmask)
    tl.store(out_ptr1 + (x0), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32 = args
    args.clear()
    assert_size_stride(primals_1, (30000, 128), (128, 1))
    assert_size_stride(primals_2, (2, 128), (128, 1))
    assert_size_stride(primals_3, (512, 128), (128, 1))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (768, 128), (128, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, 768), (768, 1))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, 768), (768, 1))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, 768), (768, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (3072, 768), (768, 1))
    assert_size_stride(primals_19, (3072, ), (1, ))
    assert_size_stride(primals_20, (768, 3072), (3072, 1))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (128, 768), (768, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (30000, 128), (128, 1))
    assert_size_stride(primals_29, (30000, ), (1, ))
    assert_size_stride(primals_30, (1, 512), (512, 1))
    assert_size_stride(primals_31, (1, 512), (512, 1))
    assert_size_stride(primals_32, (4, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 512, 128), device='cuda', dtype=torch.float32)
        buf4 = empty((4, 512, 128), device='cuda', dtype=torch.float32)
        buf5 = empty((2048, 128), device='cuda', dtype=torch.float32)
        buf398 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [embeddings, embeddings_1, embeddings_2, hidden_states, inputs_embeds, position_embeddings, token_type_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_view_0.run(primals_32, primals_1, primals_30, primals_2, primals_31, primals_3, primals_4, primals_5, buf0, buf4, buf5, buf398, 2048, 128, grid=grid(2048), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        del primals_5
        buf6 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, buf5, reinterpret_tensor(primals_6, (128, 768), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del primals_7
        buf7 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf7)
        buf8 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf8)
        buf9 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf6, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf9)
        buf10 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf7, primals_9, buf10, 1572864, grid=grid(1572864), stream=stream0)
        buf11 = reinterpret_tensor(buf7, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf7  # reuse
        # Source Nodes: [attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf8, primals_11, buf11, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf12 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf11, (48, 64, 512), (32768, 512, 1), 0), out=buf12)
        buf15 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf397 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs, attention_probs_1, attention_scores_2], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf12, buf15, buf397, 24576, 512, grid=grid(24576), stream=stream0)
        buf16 = reinterpret_tensor(buf8, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [context_layer], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf9, primals_13, buf16, 1572864, grid=grid(1572864), stream=stream0)
        buf17 = reinterpret_tensor(buf9, (48, 512, 64), (32768, 64, 1), 0); del buf9  # reuse
        # Source Nodes: [context_layer], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf16, (48, 512, 64), (32768, 64, 1), 0), out=buf17)
        buf18 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf17, buf18, 1572864, grid=grid(1572864), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (2048, 768), (768, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf18, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf19)
        buf23 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf24 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf396 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, ffn_output, layernormed_context_layer], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf6, buf19, primals_15, primals_16, primals_17, buf23, buf24, buf396, 2048, 768, grid=grid(2048), stream=stream0)
        buf25 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf24, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf25)
        buf26 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf27 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_3, add_4, ffn_output_1, ffn_output_3, mul_1, mul_2, mul_3, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf25, buf26, buf27, 6291456, grid=grid(6291456), stream=stream0)
        buf28 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf27, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf28)
        buf29 = reinterpret_tensor(buf28, (4, 512, 768), (393216, 768, 1), 0); del buf28  # reuse
        buf33 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf34 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf395 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_5, hidden_states_3, layernormed_context_layer, mixed_query_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf29, primals_21, buf23, primals_16, primals_17, primals_22, primals_23, buf33, buf34, buf395, 2048, 768, grid=grid(2048), stream=stream0)
        buf35 = reinterpret_tensor(buf29, (2048, 768), (768, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf34, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf35)
        buf36 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf34, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf36)
        buf37 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf34, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf37)
        buf38 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf35, primals_9, buf38, 1572864, grid=grid(1572864), stream=stream0)
        buf39 = reinterpret_tensor(buf35, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf35  # reuse
        # Source Nodes: [attention_scores_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf36, primals_11, buf39, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf40 = buf12; del buf12  # reuse
        # Source Nodes: [attention_scores_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf38, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf39, (48, 64, 512), (32768, 512, 1), 0), out=buf40)
        buf43 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf394 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_2, attention_probs_3, attention_scores_5], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf40, buf43, buf394, 24576, 512, grid=grid(24576), stream=stream0)
        buf44 = reinterpret_tensor(buf36, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [context_layer_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf37, primals_13, buf44, 1572864, grid=grid(1572864), stream=stream0)
        buf45 = reinterpret_tensor(buf37, (48, 512, 64), (32768, 64, 1), 0); del buf37  # reuse
        # Source Nodes: [context_layer_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf44, (48, 512, 64), (32768, 64, 1), 0), out=buf45)
        buf46 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_1], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf45, buf46, 1572864, grid=grid(1572864), stream=stream0)
        buf47 = reinterpret_tensor(buf45, (2048, 768), (768, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf46, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf47)
        buf48 = reinterpret_tensor(buf47, (4, 512, 768), (393216, 768, 1), 0); del buf47  # reuse
        buf52 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf53 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf393 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_7, ffn_output_4, hidden_states_3, layernormed_context_layer_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf48, buf33, primals_22, primals_23, primals_15, primals_16, primals_17, buf52, buf53, buf393, 2048, 768, grid=grid(2048), stream=stream0)
        buf54 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf53, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf54)
        buf55 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf56 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, add_9, ffn_output_5, ffn_output_7, mul_5, mul_6, mul_7, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf54, buf55, buf56, 6291456, grid=grid(6291456), stream=stream0)
        buf57 = reinterpret_tensor(buf48, (2048, 768), (768, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf56, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf57)
        buf58 = reinterpret_tensor(buf57, (4, 512, 768), (393216, 768, 1), 0); del buf57  # reuse
        buf62 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf63 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf392 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, hidden_states_6, layernormed_context_layer_1, mixed_query_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf58, primals_21, buf52, primals_16, primals_17, primals_22, primals_23, buf62, buf63, buf392, 2048, 768, grid=grid(2048), stream=stream0)
        buf64 = reinterpret_tensor(buf58, (2048, 768), (768, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf63, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf64)
        buf65 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf63, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf65)
        buf66 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf63, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf66)
        buf67 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf64, primals_9, buf67, 1572864, grid=grid(1572864), stream=stream0)
        buf68 = reinterpret_tensor(buf64, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf64  # reuse
        # Source Nodes: [attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf65, primals_11, buf68, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf69 = buf40; del buf40  # reuse
        # Source Nodes: [attention_scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf67, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf68, (48, 64, 512), (32768, 512, 1), 0), out=buf69)
        buf72 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf391 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_4, attention_probs_5, attention_scores_8], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf69, buf72, buf391, 24576, 512, grid=grid(24576), stream=stream0)
        buf73 = reinterpret_tensor(buf65, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf65  # reuse
        # Source Nodes: [context_layer_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf66, primals_13, buf73, 1572864, grid=grid(1572864), stream=stream0)
        buf74 = reinterpret_tensor(buf66, (48, 512, 64), (32768, 64, 1), 0); del buf66  # reuse
        # Source Nodes: [context_layer_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf73, (48, 512, 64), (32768, 64, 1), 0), out=buf74)
        buf75 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_2], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf74, buf75, 1572864, grid=grid(1572864), stream=stream0)
        buf76 = reinterpret_tensor(buf74, (2048, 768), (768, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf75, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf76)
        buf77 = reinterpret_tensor(buf76, (4, 512, 768), (393216, 768, 1), 0); del buf76  # reuse
        buf81 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf82 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf390 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_12, ffn_output_8, hidden_states_6, layernormed_context_layer_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf77, buf62, primals_22, primals_23, primals_15, primals_16, primals_17, buf81, buf82, buf390, 2048, 768, grid=grid(2048), stream=stream0)
        buf83 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf82, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf83)
        buf84 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf85 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_13, add_14, ffn_output_11, ffn_output_9, mul_10, mul_11, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf83, buf84, buf85, 6291456, grid=grid(6291456), stream=stream0)
        buf86 = reinterpret_tensor(buf77, (2048, 768), (768, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf85, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf86)
        buf87 = reinterpret_tensor(buf86, (4, 512, 768), (393216, 768, 1), 0); del buf86  # reuse
        buf91 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf92 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf389 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_15, hidden_states_9, layernormed_context_layer_2, mixed_query_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf87, primals_21, buf81, primals_16, primals_17, primals_22, primals_23, buf91, buf92, buf389, 2048, 768, grid=grid(2048), stream=stream0)
        buf93 = reinterpret_tensor(buf87, (2048, 768), (768, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf92, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf93)
        buf94 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf92, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf94)
        buf95 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf92, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf95)
        buf96 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf93, primals_9, buf96, 1572864, grid=grid(1572864), stream=stream0)
        buf97 = reinterpret_tensor(buf93, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf93  # reuse
        # Source Nodes: [attention_scores_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf94, primals_11, buf97, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf98 = buf69; del buf69  # reuse
        # Source Nodes: [attention_scores_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf97, (48, 64, 512), (32768, 512, 1), 0), out=buf98)
        buf101 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf388 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_6, attention_probs_7, attention_scores_11], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf98, buf101, buf388, 24576, 512, grid=grid(24576), stream=stream0)
        buf102 = reinterpret_tensor(buf94, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf94  # reuse
        # Source Nodes: [context_layer_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf95, primals_13, buf102, 1572864, grid=grid(1572864), stream=stream0)
        buf103 = reinterpret_tensor(buf95, (48, 512, 64), (32768, 64, 1), 0); del buf95  # reuse
        # Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf102, (48, 512, 64), (32768, 64, 1), 0), out=buf103)
        buf104 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_3], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf103, buf104, 1572864, grid=grid(1572864), stream=stream0)
        buf105 = reinterpret_tensor(buf103, (2048, 768), (768, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf104, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf105)
        buf106 = reinterpret_tensor(buf105, (4, 512, 768), (393216, 768, 1), 0); del buf105  # reuse
        buf110 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf111 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf387 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, ffn_output_12, hidden_states_9, layernormed_context_layer_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf106, buf91, primals_22, primals_23, primals_15, primals_16, primals_17, buf110, buf111, buf387, 2048, 768, grid=grid(2048), stream=stream0)
        buf112 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf111, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf112)
        buf113 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf114 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19, ffn_output_13, ffn_output_15, mul_13, mul_14, mul_15, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf112, buf113, buf114, 6291456, grid=grid(6291456), stream=stream0)
        buf115 = reinterpret_tensor(buf106, (2048, 768), (768, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf114, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf115)
        buf116 = reinterpret_tensor(buf115, (4, 512, 768), (393216, 768, 1), 0); del buf115  # reuse
        buf120 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf121 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf386 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, hidden_states_12, layernormed_context_layer_3, mixed_query_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf116, primals_21, buf110, primals_16, primals_17, primals_22, primals_23, buf120, buf121, buf386, 2048, 768, grid=grid(2048), stream=stream0)
        buf122 = reinterpret_tensor(buf116, (2048, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf121, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf122)
        buf123 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf121, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf123)
        buf124 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf121, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf124)
        buf125 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf122, primals_9, buf125, 1572864, grid=grid(1572864), stream=stream0)
        buf126 = reinterpret_tensor(buf122, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf122  # reuse
        # Source Nodes: [attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf123, primals_11, buf126, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf127 = buf98; del buf98  # reuse
        # Source Nodes: [attention_scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf126, (48, 64, 512), (32768, 512, 1), 0), out=buf127)
        buf130 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf385 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_8, attention_probs_9, attention_scores_14], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf127, buf130, buf385, 24576, 512, grid=grid(24576), stream=stream0)
        buf131 = reinterpret_tensor(buf123, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf123  # reuse
        # Source Nodes: [context_layer_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf124, primals_13, buf131, 1572864, grid=grid(1572864), stream=stream0)
        buf132 = reinterpret_tensor(buf124, (48, 512, 64), (32768, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [context_layer_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf131, (48, 512, 64), (32768, 64, 1), 0), out=buf132)
        buf133 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_4], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf132, buf133, 1572864, grid=grid(1572864), stream=stream0)
        buf134 = reinterpret_tensor(buf132, (2048, 768), (768, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf133, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf134)
        buf135 = reinterpret_tensor(buf134, (4, 512, 768), (393216, 768, 1), 0); del buf134  # reuse
        buf139 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf140 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf384 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, ffn_output_16, hidden_states_12, layernormed_context_layer_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf135, buf120, primals_22, primals_23, primals_15, primals_16, primals_17, buf139, buf140, buf384, 2048, 768, grid=grid(2048), stream=stream0)
        buf141 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf140, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf141)
        buf142 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf143 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, add_24, ffn_output_17, ffn_output_19, mul_17, mul_18, mul_19, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf141, buf142, buf143, 6291456, grid=grid(6291456), stream=stream0)
        buf144 = reinterpret_tensor(buf135, (2048, 768), (768, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf143, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf144)
        buf145 = reinterpret_tensor(buf144, (4, 512, 768), (393216, 768, 1), 0); del buf144  # reuse
        buf149 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf150 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf383 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, hidden_states_15, layernormed_context_layer_4, mixed_query_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf145, primals_21, buf139, primals_16, primals_17, primals_22, primals_23, buf149, buf150, buf383, 2048, 768, grid=grid(2048), stream=stream0)
        buf151 = reinterpret_tensor(buf145, (2048, 768), (768, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf150, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf151)
        buf152 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf150, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf152)
        buf153 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf150, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf153)
        buf154 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf151, primals_9, buf154, 1572864, grid=grid(1572864), stream=stream0)
        buf155 = reinterpret_tensor(buf151, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf151  # reuse
        # Source Nodes: [attention_scores_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf152, primals_11, buf155, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf156 = buf127; del buf127  # reuse
        # Source Nodes: [attention_scores_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf155, (48, 64, 512), (32768, 512, 1), 0), out=buf156)
        buf159 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf382 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_10, attention_probs_11, attention_scores_17], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf156, buf159, buf382, 24576, 512, grid=grid(24576), stream=stream0)
        buf160 = reinterpret_tensor(buf152, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf152  # reuse
        # Source Nodes: [context_layer_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf153, primals_13, buf160, 1572864, grid=grid(1572864), stream=stream0)
        buf161 = reinterpret_tensor(buf153, (48, 512, 64), (32768, 64, 1), 0); del buf153  # reuse
        # Source Nodes: [context_layer_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf159, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf160, (48, 512, 64), (32768, 64, 1), 0), out=buf161)
        buf162 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_5], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf161, buf162, 1572864, grid=grid(1572864), stream=stream0)
        buf163 = reinterpret_tensor(buf161, (2048, 768), (768, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf162, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf163)
        buf164 = reinterpret_tensor(buf163, (4, 512, 768), (393216, 768, 1), 0); del buf163  # reuse
        buf168 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf169 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf381 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_27, ffn_output_20, hidden_states_15, layernormed_context_layer_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf164, buf149, primals_22, primals_23, primals_15, primals_16, primals_17, buf168, buf169, buf381, 2048, 768, grid=grid(2048), stream=stream0)
        buf170 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf169, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf170)
        buf171 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf172 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, add_29, ffn_output_21, ffn_output_23, mul_21, mul_22, mul_23, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf170, buf171, buf172, 6291456, grid=grid(6291456), stream=stream0)
        buf173 = reinterpret_tensor(buf164, (2048, 768), (768, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf172, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf173)
        buf174 = reinterpret_tensor(buf173, (4, 512, 768), (393216, 768, 1), 0); del buf173  # reuse
        buf178 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf179 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf380 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, hidden_states_18, layernormed_context_layer_5, mixed_query_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf174, primals_21, buf168, primals_16, primals_17, primals_22, primals_23, buf178, buf179, buf380, 2048, 768, grid=grid(2048), stream=stream0)
        buf180 = reinterpret_tensor(buf174, (2048, 768), (768, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf179, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf180)
        buf181 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf179, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf181)
        buf182 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf179, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf182)
        buf183 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf180, primals_9, buf183, 1572864, grid=grid(1572864), stream=stream0)
        buf184 = reinterpret_tensor(buf180, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf180  # reuse
        # Source Nodes: [attention_scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf181, primals_11, buf184, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf185 = buf156; del buf156  # reuse
        # Source Nodes: [attention_scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf184, (48, 64, 512), (32768, 512, 1), 0), out=buf185)
        buf188 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf379 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_12, attention_probs_13, attention_scores_20], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf185, buf188, buf379, 24576, 512, grid=grid(24576), stream=stream0)
        buf189 = reinterpret_tensor(buf181, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf181  # reuse
        # Source Nodes: [context_layer_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf182, primals_13, buf189, 1572864, grid=grid(1572864), stream=stream0)
        buf190 = reinterpret_tensor(buf182, (48, 512, 64), (32768, 64, 1), 0); del buf182  # reuse
        # Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf188, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf189, (48, 512, 64), (32768, 64, 1), 0), out=buf190)
        buf191 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_6], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf190, buf191, 1572864, grid=grid(1572864), stream=stream0)
        buf192 = reinterpret_tensor(buf190, (2048, 768), (768, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf191, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf192)
        buf193 = reinterpret_tensor(buf192, (4, 512, 768), (393216, 768, 1), 0); del buf192  # reuse
        buf197 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf198 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf378 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_32, ffn_output_24, hidden_states_18, layernormed_context_layer_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf193, buf178, primals_22, primals_23, primals_15, primals_16, primals_17, buf197, buf198, buf378, 2048, 768, grid=grid(2048), stream=stream0)
        buf199 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf198, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf199)
        buf200 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf201 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_33, add_34, ffn_output_25, ffn_output_27, mul_25, mul_26, mul_27, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf199, buf200, buf201, 6291456, grid=grid(6291456), stream=stream0)
        buf202 = reinterpret_tensor(buf193, (2048, 768), (768, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf201, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf202)
        buf203 = reinterpret_tensor(buf202, (4, 512, 768), (393216, 768, 1), 0); del buf202  # reuse
        buf207 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf208 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf377 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_35, hidden_states_21, layernormed_context_layer_6, mixed_query_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf203, primals_21, buf197, primals_16, primals_17, primals_22, primals_23, buf207, buf208, buf377, 2048, 768, grid=grid(2048), stream=stream0)
        buf209 = reinterpret_tensor(buf203, (2048, 768), (768, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf208, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf209)
        buf210 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf208, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf210)
        buf211 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf208, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf211)
        buf212 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf209, primals_9, buf212, 1572864, grid=grid(1572864), stream=stream0)
        buf213 = reinterpret_tensor(buf209, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf209  # reuse
        # Source Nodes: [attention_scores_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf210, primals_11, buf213, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf214 = buf185; del buf185  # reuse
        # Source Nodes: [attention_scores_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf212, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf213, (48, 64, 512), (32768, 512, 1), 0), out=buf214)
        buf217 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf376 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_14, attention_probs_15, attention_scores_23], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf214, buf217, buf376, 24576, 512, grid=grid(24576), stream=stream0)
        buf218 = reinterpret_tensor(buf210, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf210  # reuse
        # Source Nodes: [context_layer_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf211, primals_13, buf218, 1572864, grid=grid(1572864), stream=stream0)
        buf219 = reinterpret_tensor(buf211, (48, 512, 64), (32768, 64, 1), 0); del buf211  # reuse
        # Source Nodes: [context_layer_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf217, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf218, (48, 512, 64), (32768, 64, 1), 0), out=buf219)
        buf220 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_7], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf219, buf220, 1572864, grid=grid(1572864), stream=stream0)
        buf221 = reinterpret_tensor(buf219, (2048, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf220, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf221)
        buf222 = reinterpret_tensor(buf221, (4, 512, 768), (393216, 768, 1), 0); del buf221  # reuse
        buf226 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf227 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf375 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_37, ffn_output_28, hidden_states_21, layernormed_context_layer_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf222, buf207, primals_22, primals_23, primals_15, primals_16, primals_17, buf226, buf227, buf375, 2048, 768, grid=grid(2048), stream=stream0)
        buf228 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf227, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf228)
        buf229 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf230 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, add_39, ffn_output_29, ffn_output_31, mul_29, mul_30, mul_31, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf228, buf229, buf230, 6291456, grid=grid(6291456), stream=stream0)
        buf231 = reinterpret_tensor(buf222, (2048, 768), (768, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf230, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf231)
        buf232 = reinterpret_tensor(buf231, (4, 512, 768), (393216, 768, 1), 0); del buf231  # reuse
        buf236 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf237 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf374 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_40, hidden_states_24, layernormed_context_layer_7, mixed_query_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf232, primals_21, buf226, primals_16, primals_17, primals_22, primals_23, buf236, buf237, buf374, 2048, 768, grid=grid(2048), stream=stream0)
        buf238 = reinterpret_tensor(buf232, (2048, 768), (768, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf237, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf238)
        buf239 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf237, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf239)
        buf240 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf237, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf240)
        buf241 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf238, primals_9, buf241, 1572864, grid=grid(1572864), stream=stream0)
        buf242 = reinterpret_tensor(buf238, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf238  # reuse
        # Source Nodes: [attention_scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf239, primals_11, buf242, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf243 = buf214; del buf214  # reuse
        # Source Nodes: [attention_scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf242, (48, 64, 512), (32768, 512, 1), 0), out=buf243)
        buf246 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf373 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_16, attention_probs_17, attention_scores_26], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf243, buf246, buf373, 24576, 512, grid=grid(24576), stream=stream0)
        buf247 = reinterpret_tensor(buf239, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf239  # reuse
        # Source Nodes: [context_layer_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf240, primals_13, buf247, 1572864, grid=grid(1572864), stream=stream0)
        buf248 = reinterpret_tensor(buf240, (48, 512, 64), (32768, 64, 1), 0); del buf240  # reuse
        # Source Nodes: [context_layer_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf247, (48, 512, 64), (32768, 64, 1), 0), out=buf248)
        buf249 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_8], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf248, buf249, 1572864, grid=grid(1572864), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (2048, 768), (768, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf249, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf250)
        buf251 = reinterpret_tensor(buf250, (4, 512, 768), (393216, 768, 1), 0); del buf250  # reuse
        buf255 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf256 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf372 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, ffn_output_32, hidden_states_24, layernormed_context_layer_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf251, buf236, primals_22, primals_23, primals_15, primals_16, primals_17, buf255, buf256, buf372, 2048, 768, grid=grid(2048), stream=stream0)
        buf257 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf256, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf257)
        buf258 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf259 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_43, add_44, ffn_output_33, ffn_output_35, mul_33, mul_34, mul_35, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf257, buf258, buf259, 6291456, grid=grid(6291456), stream=stream0)
        buf260 = reinterpret_tensor(buf251, (2048, 768), (768, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf259, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf260)
        buf261 = reinterpret_tensor(buf260, (4, 512, 768), (393216, 768, 1), 0); del buf260  # reuse
        buf265 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf266 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf371 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_45, hidden_states_27, layernormed_context_layer_8, mixed_query_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf261, primals_21, buf255, primals_16, primals_17, primals_22, primals_23, buf265, buf266, buf371, 2048, 768, grid=grid(2048), stream=stream0)
        buf267 = reinterpret_tensor(buf261, (2048, 768), (768, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf266, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf267)
        buf268 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf266, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf268)
        buf269 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf266, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf269)
        buf270 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf267, primals_9, buf270, 1572864, grid=grid(1572864), stream=stream0)
        buf271 = reinterpret_tensor(buf267, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf267  # reuse
        # Source Nodes: [attention_scores_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf268, primals_11, buf271, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf272 = buf243; del buf243  # reuse
        # Source Nodes: [attention_scores_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf271, (48, 64, 512), (32768, 512, 1), 0), out=buf272)
        buf275 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf370 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_18, attention_probs_19, attention_scores_29], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf272, buf275, buf370, 24576, 512, grid=grid(24576), stream=stream0)
        buf276 = reinterpret_tensor(buf268, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf268  # reuse
        # Source Nodes: [context_layer_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf269, primals_13, buf276, 1572864, grid=grid(1572864), stream=stream0)
        buf277 = reinterpret_tensor(buf269, (48, 512, 64), (32768, 64, 1), 0); del buf269  # reuse
        # Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf275, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf276, (48, 512, 64), (32768, 64, 1), 0), out=buf277)
        buf278 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_9], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf277, buf278, 1572864, grid=grid(1572864), stream=stream0)
        buf279 = reinterpret_tensor(buf277, (2048, 768), (768, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf278, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf279)
        buf280 = reinterpret_tensor(buf279, (4, 512, 768), (393216, 768, 1), 0); del buf279  # reuse
        buf284 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf285 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf369 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_47, ffn_output_36, hidden_states_27, layernormed_context_layer_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf280, buf265, primals_22, primals_23, primals_15, primals_16, primals_17, buf284, buf285, buf369, 2048, 768, grid=grid(2048), stream=stream0)
        buf286 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf285, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf286)
        buf287 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf288 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_48, add_49, ffn_output_37, ffn_output_39, mul_37, mul_38, mul_39, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf286, buf287, buf288, 6291456, grid=grid(6291456), stream=stream0)
        buf289 = reinterpret_tensor(buf280, (2048, 768), (768, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf288, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf289)
        buf290 = reinterpret_tensor(buf289, (4, 512, 768), (393216, 768, 1), 0); del buf289  # reuse
        buf294 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf295 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf368 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_50, hidden_states_30, layernormed_context_layer_9, mixed_query_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf290, primals_21, buf284, primals_16, primals_17, primals_22, primals_23, buf294, buf295, buf368, 2048, 768, grid=grid(2048), stream=stream0)
        buf296 = reinterpret_tensor(buf290, (2048, 768), (768, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf295, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf296)
        buf297 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf295, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf297)
        buf298 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf295, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf298)
        buf299 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf296, primals_9, buf299, 1572864, grid=grid(1572864), stream=stream0)
        buf300 = reinterpret_tensor(buf296, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf296  # reuse
        # Source Nodes: [attention_scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf297, primals_11, buf300, 3072, 512, grid=grid(3072, 512), stream=stream0)
        buf301 = buf272; del buf272  # reuse
        # Source Nodes: [attention_scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf299, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf300, (48, 64, 512), (32768, 512, 1), 0), out=buf301)
        buf304 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf367 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_20, attention_probs_21, attention_scores_32], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf301, buf304, buf367, 24576, 512, grid=grid(24576), stream=stream0)
        buf305 = reinterpret_tensor(buf297, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf297  # reuse
        # Source Nodes: [context_layer_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf298, primals_13, buf305, 1572864, grid=grid(1572864), stream=stream0)
        buf306 = reinterpret_tensor(buf298, (48, 512, 64), (32768, 64, 1), 0); del buf298  # reuse
        # Source Nodes: [context_layer_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf305, (48, 512, 64), (32768, 64, 1), 0), out=buf306)
        buf307 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_10], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf306, buf307, 1572864, grid=grid(1572864), stream=stream0)
        buf308 = reinterpret_tensor(buf306, (2048, 768), (768, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf307, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf308)
        buf309 = reinterpret_tensor(buf308, (4, 512, 768), (393216, 768, 1), 0); del buf308  # reuse
        buf313 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf314 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf366 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_52, ffn_output_40, hidden_states_30, layernormed_context_layer_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf309, buf294, primals_22, primals_23, primals_15, primals_16, primals_17, buf313, buf314, buf366, 2048, 768, grid=grid(2048), stream=stream0)
        buf315 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf314, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf315)
        buf316 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf317 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_53, add_54, ffn_output_41, ffn_output_43, mul_41, mul_42, mul_43, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf315, buf316, buf317, 6291456, grid=grid(6291456), stream=stream0)
        buf318 = reinterpret_tensor(buf309, (2048, 768), (768, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf317, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf318)
        buf319 = reinterpret_tensor(buf318, (4, 512, 768), (393216, 768, 1), 0); del buf318  # reuse
        buf323 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf324 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf365 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_55, hidden_states_33, layernormed_context_layer_10, mixed_query_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf319, primals_21, buf313, primals_16, primals_17, primals_22, primals_23, buf323, buf324, buf365, 2048, 768, grid=grid(2048), stream=stream0)
        buf325 = reinterpret_tensor(buf319, (2048, 768), (768, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf324, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf325)
        buf326 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf324, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf326)
        buf327 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf324, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf327)
        buf328 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_scores_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf325, primals_9, buf328, 1572864, grid=grid(1572864), stream=stream0)
        del primals_9
        buf329 = reinterpret_tensor(buf325, (4, 12, 64, 512), (393216, 32768, 512, 1), 0); del buf325  # reuse
        # Source Nodes: [attention_scores_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf326, primals_11, buf329, 3072, 512, grid=grid(3072, 512), stream=stream0)
        del primals_11
        buf330 = buf301; del buf301  # reuse
        # Source Nodes: [attention_scores_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf328, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf329, (48, 64, 512), (32768, 512, 1), 0), out=buf330)
        buf333 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        buf364 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attention_probs_22, attention_probs_23, attention_scores_35], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach]
        triton_per_fused__softmax_add_clone_detach_3.run(buf330, buf333, buf364, 24576, 512, grid=grid(24576), stream=stream0)
        del buf330
        buf334 = reinterpret_tensor(buf326, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf326  # reuse
        # Source Nodes: [context_layer_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf327, primals_13, buf334, 1572864, grid=grid(1572864), stream=stream0)
        del primals_13
        buf335 = reinterpret_tensor(buf327, (48, 512, 64), (32768, 64, 1), 0); del buf327  # reuse
        # Source Nodes: [context_layer_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf333, (48, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf334, (48, 512, 64), (32768, 64, 1), 0), out=buf335)
        buf336 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [projected_context_layer_11], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf335, buf336, 1572864, grid=grid(1572864), stream=stream0)
        buf337 = reinterpret_tensor(buf335, (2048, 768), (768, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf336, reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), out=buf337)
        buf338 = reinterpret_tensor(buf337, (4, 512, 768), (393216, 768, 1), 0); del buf337  # reuse
        buf342 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf343 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf363 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_57, ffn_output_44, hidden_states_33, layernormed_context_layer_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf338, buf323, primals_22, primals_23, primals_15, primals_16, primals_17, buf342, buf343, buf363, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_15
        buf344 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [ffn_output_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf343, reinterpret_tensor(primals_18, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf344)
        del primals_19
        buf345 = empty((4, 512, 3072), device='cuda', dtype=torch.float32)
        buf346 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_58, add_59, ffn_output_45, ffn_output_47, mul_45, mul_46, mul_47, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh, aten.view]
        triton_poi_fused_add_mul_pow_tanh_view_6.run(buf344, buf345, buf346, 6291456, grid=grid(6291456), stream=stream0)
        buf347 = reinterpret_tensor(buf338, (2048, 768), (768, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf346, reinterpret_tensor(primals_20, (3072, 768), (1, 3072), 0), out=buf347)
        buf348 = reinterpret_tensor(buf347, (4, 512, 768), (393216, 768, 1), 0); del buf347  # reuse
        buf352 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf353 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf362 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_60, hidden_states_37, layernormed_context_layer_11, sequence_outputs], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf348, primals_21, buf342, primals_16, primals_17, primals_22, primals_23, buf352, buf353, buf362, 2048, 768, grid=grid(2048), stream=stream0)
        del buf348
        del primals_17
        del primals_21
        del primals_23
        buf354 = reinterpret_tensor(buf0, (2048, 128), (128, 1), 0); del buf0  # reuse
        # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, buf353, reinterpret_tensor(primals_24, (768, 128), (1, 768), 0), alpha=1, beta=1, out=buf354)
        del primals_25
        buf355 = empty((4, 512, 128), device='cuda', dtype=torch.float32)
        buf356 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        buf357 = empty_strided((4, 512, 1), (512, 1, 2048), device='cuda', dtype=torch.float32)
        buf359 = reinterpret_tensor(buf357, (4, 512, 1), (512, 1, 1), 0); del buf357  # reuse
        buf360 = empty((2048, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_61, add_62, hidden_states_38, hidden_states_39, mul_49, mul_50, mul_51, pow_13, prediction_scores, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh, aten.view]
        triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_9.run(buf359, buf354, primals_26, primals_27, buf355, buf356, buf360, 2048, 128, grid=grid(2048), stream=stream0)
        del primals_27
        buf361 = empty((2048, 30000), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_scores], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, buf360, reinterpret_tensor(primals_28, (128, 30000), (1, 128), 0), alpha=1, beta=1, out=buf361)
        del primals_29
        return (reinterpret_tensor(buf361, (4, 512, 30000), (15360000, 30000, 1), 0), primals_4, primals_16, primals_22, primals_26, primals_32, reinterpret_tensor(primals_30, (4, 512), (0, 1), 0), primals_31, buf4, buf5, reinterpret_tensor(buf6, (2048, 768), (768, 1), 0), buf18, buf23, buf24, buf25, buf26, buf27, buf33, buf34, buf46, buf52, buf53, buf54, buf55, buf56, buf62, buf63, buf75, buf81, buf82, buf83, buf84, buf85, buf91, buf92, buf104, buf110, buf111, buf112, buf113, buf114, buf120, buf121, buf133, buf139, buf140, buf141, buf142, buf143, buf149, buf150, buf162, buf168, buf169, buf170, buf171, buf172, buf178, buf179, buf191, buf197, buf198, buf199, buf200, buf201, buf207, buf208, buf220, buf226, buf227, buf228, buf229, buf230, buf236, buf237, buf249, buf255, buf256, buf257, buf258, buf259, buf265, buf266, buf278, buf284, buf285, buf286, buf287, buf288, buf294, buf295, buf307, buf313, buf314, buf315, buf316, buf317, buf323, buf324, buf336, buf342, buf343, buf344, buf345, buf346, buf352, buf353, buf354, buf355, buf356, buf359, buf360, reinterpret_tensor(primals_28, (30000, 128), (128, 1), 0), reinterpret_tensor(primals_24, (128, 768), (768, 1), 0), buf362, reinterpret_tensor(primals_20, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_18, (3072, 768), (768, 1), 0), buf363, reinterpret_tensor(primals_14, (768, 768), (768, 1), 0), reinterpret_tensor(buf333, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf334, (48, 64, 512), (32768, 1, 64), 0), buf364, reinterpret_tensor(buf328, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf329, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_12, (768, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (768, 1), 0), buf365, buf366, reinterpret_tensor(buf304, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf305, (48, 64, 512), (32768, 1, 64), 0), buf367, reinterpret_tensor(buf299, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf300, (48, 512, 64), (32768, 1, 512), 0), buf368, buf369, reinterpret_tensor(buf275, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf276, (48, 64, 512), (32768, 1, 64), 0), buf370, reinterpret_tensor(buf270, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf271, (48, 512, 64), (32768, 1, 512), 0), buf371, buf372, reinterpret_tensor(buf246, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf247, (48, 64, 512), (32768, 1, 64), 0), buf373, reinterpret_tensor(buf241, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf242, (48, 512, 64), (32768, 1, 512), 0), buf374, buf375, reinterpret_tensor(buf217, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf218, (48, 64, 512), (32768, 1, 64), 0), buf376, reinterpret_tensor(buf212, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf213, (48, 512, 64), (32768, 1, 512), 0), buf377, buf378, reinterpret_tensor(buf188, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf189, (48, 64, 512), (32768, 1, 64), 0), buf379, reinterpret_tensor(buf183, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf184, (48, 512, 64), (32768, 1, 512), 0), buf380, buf381, reinterpret_tensor(buf159, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf160, (48, 64, 512), (32768, 1, 64), 0), buf382, reinterpret_tensor(buf154, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf155, (48, 512, 64), (32768, 1, 512), 0), buf383, buf384, reinterpret_tensor(buf130, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf131, (48, 64, 512), (32768, 1, 64), 0), buf385, reinterpret_tensor(buf125, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf126, (48, 512, 64), (32768, 1, 512), 0), buf386, buf387, reinterpret_tensor(buf101, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf102, (48, 64, 512), (32768, 1, 64), 0), buf388, reinterpret_tensor(buf96, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf97, (48, 512, 64), (32768, 1, 512), 0), buf389, buf390, reinterpret_tensor(buf72, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf73, (48, 64, 512), (32768, 1, 64), 0), buf391, reinterpret_tensor(buf67, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf68, (48, 512, 64), (32768, 1, 512), 0), buf392, buf393, reinterpret_tensor(buf43, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf44, (48, 64, 512), (32768, 1, 64), 0), buf394, reinterpret_tensor(buf38, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf39, (48, 512, 64), (32768, 1, 512), 0), buf395, buf396, reinterpret_tensor(buf15, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf16, (48, 64, 512), (32768, 1, 64), 0), buf397, reinterpret_tensor(buf10, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf11, (48, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_6, (768, 128), (128, 1), 0), buf398, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((30000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((30000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_31 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_32 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Albert', benchmark_compiled_module)
