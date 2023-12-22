
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


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlkazf2nmbvx7x5sqgkauvfsybec2yladhmt4bwvmofk32pmhwr.py
# Source Nodes: [position_ids_1], Original ATen: [aten.view]
# position_ids_1 => view_1
triton_poi_fused_view_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ih/cihkyycf6lczecopepixwzgeyn3uarg75sg7xu2hkome3gcv6ls5.py
# Source Nodes: [add, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding]
# add => add
# inputs_embeds => embedding
# position_embeds => embedding_1
triton_poi_fused_add_embedding_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x0 = xindex % 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp1 = tmp0 + 50257
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 50257), "index out of bounds: 0 <= tmp3 < 50257")
    tmp4 = tl.load(in_ptr1 + (x0 + (768*tmp3)), None)
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbvpyfkcv2ouj4y7pp4w7r6r7js2sluifmnjvzngd262za2nm5v.py
# Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
triton_per_fused_native_layer_norm_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 768.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp22 / tmp18
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czmqm2x7fwp3b5wlgcjzoy45wjzp5smzke4al5mgjjt3o5gbgtqg.py
# Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
# attn_weights_1 => div
# attn_weights_2 => where
# attn_weights_3 => amax, div_1, exp, sub_1, sum_1
# full => full_default
# mask_value => full_default_1
triton_per_fused__softmax__to_copy_div_full_where_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_div_full_where_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x3)), rmask, other=0.0)
    tmp2 = 8.0
    tmp3 = tmp1 / tmp2
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp0, tmp3, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, float("-inf"))
    tmp9 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp8, 0))
    tmp10 = tmp5 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tmp11 / tmp15
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp16, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbo3t6hsb3qcy6dnas3b6gm7h2vcjuuipjkeoejm5s3g7g245q2.py
# Source Nodes: [tensor_3], Original ATen: [aten.clone]
# tensor_3 => clone
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cru3nbuwz2wbucz3jvy3pdlyaedtka7wow6kqa6nvh45zq35ay47.py
# Source Nodes: [hidden_states_2, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states_2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# residual_1 => add_3
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp22 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/rq/crqlymohttcd5frb5oaxp4omonzplgh2sgfucvhid7l3kl7buwwf.py
# Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_2 => add_6
# add_3 => add_7
# hidden_states_4 => mul_7
# mul => mul_4
# mul_1 => mul_5
# mul_2 => mul_6
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
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


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdnafwi7wcp22nmuufu7qeet7cnke6srwn5rvfv7te4wdowxapo.py
# Source Nodes: [hidden_states_8, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states_8 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
# residual_1 => add_3
# residual_2 => add_8
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp24 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/yw/cywjjxv3pd6hgg5giojo36qrqxrjnaku7ss6pqqwa3mgsmquctel.py
# Source Nodes: [hidden_states_10, residual_1, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states_10 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
# residual_1 => add_3
# residual_2 => add_8
# residual_3 => add_11
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
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
    tmp26 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/66/c66w2gsblxrgav5c4jou6yqq33qskzvdehlokdbpasgp4zzpodzt.py
# Source Nodes: [hidden_states_16, residual_1, residual_2, residual_3, residual_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states_16 => add_17, add_18, mul_16, mul_17, rsqrt_4, sub_6, var_mean_4
# residual_1 => add_3
# residual_2 => add_8
# residual_3 => add_11
# residual_4 => add_16
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 + tmp7
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
    tmp28 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45twonrp2f322pefnuhxm5vqgpganqijxygo3xiyhj7q4473e5b.py
# Source Nodes: [argmax, eq, long, sub], Original ATen: [aten._to_copy, aten.argmax, aten.eq, aten.sub]
# argmax => argmax
# eq => eq
# long => convert_element_type_12
# sub => sub_37
triton_red_fused__to_copy_argmax_eq_sub_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_argmax_eq_sub_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], -9223372036854775808, tl.int64)
    _tmp5_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 == tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        _tmp5_next, _tmp5_index_next = triton_helpers.maximum_with_index(
            _tmp5, _tmp5_index, tmp4, rindex
        )
        _tmp5 = tl.where(rmask, _tmp5_next, _tmp5)
        _tmp5_index = tl.where(rmask, _tmp5_index_next, _tmp5_index)
    _, tmp5_tmp = triton_helpers.max_with_index(_tmp5, _tmp5_index, 1)
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp5 - tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zgdspaoldgg2jinbwngios75ccnd24wz6uk4qcspqdjcdoiwkm.py
# Source Nodes: [arange_1], Original ATen: [aten.arange]
# arange_1 => full_default_24
triton_poi_fused_arange_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6g5rkkrceycasrr4n5cca4ypoj3y3toi7hpczpuhykmuvt6xhgf.py
# Source Nodes: [pooled_logits], Original ATen: [aten.index]
# pooled_logits => index
triton_poi_fused_index_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tmp1 + 1024
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert((0 <= tmp4) & (tmp4 < 1024), "index out of bounds: 0 <= tmp4 < 1024")
    tmp5 = tl.load(in_ptr1 + (x0 + (2*tmp4)), xmask)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162 = args
    args.clear()
    assert_size_stride(primals_1, (2304, ), (1, ))
    assert_size_stride(primals_2, (768, 2304), (2304, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, 768), (768, 1))
    assert_size_stride(primals_5, (3072, ), (1, ))
    assert_size_stride(primals_6, (768, 3072), (3072, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (3072, 768), (768, 1))
    assert_size_stride(primals_9, (2304, ), (1, ))
    assert_size_stride(primals_10, (768, 2304), (2304, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, 768), (768, 1))
    assert_size_stride(primals_13, (3072, ), (1, ))
    assert_size_stride(primals_14, (768, 3072), (3072, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (3072, 768), (768, 1))
    assert_size_stride(primals_17, (2304, ), (1, ))
    assert_size_stride(primals_18, (768, 2304), (2304, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, 768), (768, 1))
    assert_size_stride(primals_21, (3072, ), (1, ))
    assert_size_stride(primals_22, (768, 3072), (3072, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (3072, 768), (768, 1))
    assert_size_stride(primals_25, (2304, ), (1, ))
    assert_size_stride(primals_26, (768, 2304), (2304, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, 768), (768, 1))
    assert_size_stride(primals_29, (3072, ), (1, ))
    assert_size_stride(primals_30, (768, 3072), (3072, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (3072, 768), (768, 1))
    assert_size_stride(primals_33, (2304, ), (1, ))
    assert_size_stride(primals_34, (768, 2304), (2304, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, 768), (768, 1))
    assert_size_stride(primals_37, (3072, ), (1, ))
    assert_size_stride(primals_38, (768, 3072), (3072, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (3072, 768), (768, 1))
    assert_size_stride(primals_41, (2304, ), (1, ))
    assert_size_stride(primals_42, (768, 2304), (2304, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, 768), (768, 1))
    assert_size_stride(primals_45, (3072, ), (1, ))
    assert_size_stride(primals_46, (768, 3072), (3072, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (3072, 768), (768, 1))
    assert_size_stride(primals_49, (2304, ), (1, ))
    assert_size_stride(primals_50, (768, 2304), (2304, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, 768), (768, 1))
    assert_size_stride(primals_53, (3072, ), (1, ))
    assert_size_stride(primals_54, (768, 3072), (3072, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (3072, 768), (768, 1))
    assert_size_stride(primals_57, (2304, ), (1, ))
    assert_size_stride(primals_58, (768, 2304), (2304, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, 768), (768, 1))
    assert_size_stride(primals_61, (3072, ), (1, ))
    assert_size_stride(primals_62, (768, 3072), (3072, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (3072, 768), (768, 1))
    assert_size_stride(primals_65, (2304, ), (1, ))
    assert_size_stride(primals_66, (768, 2304), (2304, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, 768), (768, 1))
    assert_size_stride(primals_69, (3072, ), (1, ))
    assert_size_stride(primals_70, (768, 3072), (3072, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (3072, 768), (768, 1))
    assert_size_stride(primals_73, (2304, ), (1, ))
    assert_size_stride(primals_74, (768, 2304), (2304, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, 768), (768, 1))
    assert_size_stride(primals_77, (3072, ), (1, ))
    assert_size_stride(primals_78, (768, 3072), (3072, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (3072, 768), (768, 1))
    assert_size_stride(primals_81, (2304, ), (1, ))
    assert_size_stride(primals_82, (768, 2304), (2304, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, 768), (768, 1))
    assert_size_stride(primals_85, (3072, ), (1, ))
    assert_size_stride(primals_86, (768, 3072), (3072, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (3072, 768), (768, 1))
    assert_size_stride(primals_89, (2304, ), (1, ))
    assert_size_stride(primals_90, (768, 2304), (2304, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, 768), (768, 1))
    assert_size_stride(primals_93, (3072, ), (1, ))
    assert_size_stride(primals_94, (768, 3072), (3072, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (3072, 768), (768, 1))
    assert_size_stride(primals_97, (50257, 768), (768, 1))
    assert_size_stride(primals_98, (1024, 768), (768, 1))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (2, 768), (768, 1))
    assert_size_stride(primals_150, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_151, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_152, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_153, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_154, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_155, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_156, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_157, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_158, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_159, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_160, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_161, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_162, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [position_ids_1], Original ATen: [aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_view_0.run(buf0, 1024, grid=grid(1024), stream=stream0)
        buf1 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding]
        triton_poi_fused_add_embedding_1.run(primals_162, primals_97, primals_98, buf1, 786432, grid=grid(786432), stream=stream0)
        del primals_97
        del primals_98
        # Source Nodes: [add, inputs_embeds, position_embeds, residual], Original ATen: [aten.add, aten.embedding, aten.native_dropout]
        buf2 = aten.native_dropout(buf1, 0.1, True)
        buf3 = buf2[0]
        buf4 = buf2[1]
        del buf2
        buf8 = buf1; del buf1  # reuse
        buf9 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf417 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_2.run(buf3, primals_99, primals_100, buf8, buf9, buf417, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_100
        buf10 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1, reinterpret_tensor(buf9, (1024, 768), (768, 1), 0), primals_2, alpha=1, beta=1, out=buf10)
        del primals_1
        buf11 = empty((12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf10, (12, 64, 1024), (64, 1, 2304), 768), out=buf11)
        buf14 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_150, buf11, buf14, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, attn_weights_6, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf15 = aten.native_dropout(buf14, 0.1, True)
        buf16 = buf15[0]
        buf17 = buf15[1]
        del buf15
        buf18 = empty((12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf10, (12, 1024, 64), (64, 2304, 1), 1536), out=buf18)
        buf19 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf18, buf19, 786432, grid=grid(786432), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (1024, 768), (768, 1), 0); del buf18  # reuse
        # Source Nodes: [x_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_3, reinterpret_tensor(buf19, (1024, 768), (768, 1), 0), primals_4, alpha=1, beta=1, out=buf20)
        del primals_3
        # Source Nodes: [attn_output_4], Original ATen: [aten.native_dropout]
        buf21 = aten.native_dropout(reinterpret_tensor(buf20, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf27 = reinterpret_tensor(buf20, (1, 1024, 768), (786432, 768, 1), 0); del buf20  # reuse
        buf28 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf416 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_2, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf22, buf3, primals_101, primals_102, buf27, buf28, buf416, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_102
        buf29 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf28, (1024, 768), (768, 1), 0), primals_6, alpha=1, beta=1, out=buf29)
        del primals_5
        buf30 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf31 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf29, buf30, buf31, 3145728, grid=grid(3145728), stream=stream0)
        buf32 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, reinterpret_tensor(buf31, (1024, 3072), (3072, 1), 0), primals_8, alpha=1, beta=1, out=buf32)
        del primals_7
        # Source Nodes: [feed_forward_hidden_states], Original ATen: [aten.native_dropout]
        buf33 = aten.native_dropout(reinterpret_tensor(buf32, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf39 = reinterpret_tensor(buf32, (1, 1024, 768), (786432, 768, 1), 0); del buf32  # reuse
        buf40 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf415 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf22, buf3, buf34, primals_103, primals_104, buf39, buf40, buf415, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_104
        buf41 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf40, (1024, 768), (768, 1), 0), primals_10, alpha=1, beta=1, out=buf41)
        del primals_9
        buf42 = buf11; del buf11  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf41, (12, 64, 1024), (64, 1, 2304), 768), out=buf42)
        buf45 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_10, attn_weights_8, attn_weights_9, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_151, buf42, buf45, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_10, attn_weights_13, attn_weights_8, attn_weights_9, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf46 = aten.native_dropout(buf45, 0.1, True)
        buf47 = buf46[0]
        buf48 = buf46[1]
        del buf46
        buf49 = empty((12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf47, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf41, (12, 1024, 64), (64, 2304, 1), 1536), out=buf49)
        buf50 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf49, buf50, 786432, grid=grid(786432), stream=stream0)
        buf51 = reinterpret_tensor(buf49, (1024, 768), (768, 1), 0); del buf49  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, reinterpret_tensor(buf50, (1024, 768), (768, 1), 0), primals_12, alpha=1, beta=1, out=buf51)
        del primals_11
        # Source Nodes: [attn_output_10], Original ATen: [aten.native_dropout]
        buf52 = aten.native_dropout(reinterpret_tensor(buf51, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf58 = reinterpret_tensor(buf51, (1, 1024, 768), (786432, 768, 1), 0); del buf51  # reuse
        buf59 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf414 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_10, residual_1, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf53, buf22, buf3, buf34, primals_105, primals_106, buf58, buf59, buf414, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_106
        buf60 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf59, (1024, 768), (768, 1), 0), primals_14, alpha=1, beta=1, out=buf60)
        del primals_13
        buf61 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf62 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, add_7, hidden_states_12, mul_4, mul_5, mul_6, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf60, buf61, buf62, 3145728, grid=grid(3145728), stream=stream0)
        buf63 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_15, reinterpret_tensor(buf62, (1024, 3072), (3072, 1), 0), primals_16, alpha=1, beta=1, out=buf63)
        del primals_15
        # Source Nodes: [feed_forward_hidden_states_1], Original ATen: [aten.native_dropout]
        buf64 = aten.native_dropout(reinterpret_tensor(buf63, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf65; del buf65  # reuse
        buf71 = reinterpret_tensor(buf63, (1, 1024, 768), (786432, 768, 1), 0); del buf63  # reuse
        buf72 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf413 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_16, residual_1, residual_2, residual_3, residual_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf67, buf53, buf22, buf3, buf34, primals_107, primals_108, buf71, buf72, buf413, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_108
        buf73 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, reinterpret_tensor(buf72, (1024, 768), (768, 1), 0), primals_18, alpha=1, beta=1, out=buf73)
        del primals_17
        buf74 = buf42; del buf42  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf73, (12, 64, 1024), (64, 1, 2304), 768), out=buf74)
        buf77 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_15, attn_weights_16, attn_weights_17, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_152, buf74, buf77, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_15, attn_weights_16, attn_weights_17, attn_weights_20, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf78 = aten.native_dropout(buf77, 0.1, True)
        buf79 = buf78[0]
        buf80 = buf78[1]
        del buf78
        buf81 = reinterpret_tensor(buf53, (12, 1024, 64), (65536, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf73, (12, 1024, 64), (64, 2304, 1), 1536), out=buf81)
        buf82 = reinterpret_tensor(buf34, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [tensor_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf81, buf82, 786432, grid=grid(786432), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (1024, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, reinterpret_tensor(buf82, (1024, 768), (768, 1), 0), primals_20, alpha=1, beta=1, out=buf83)
        del primals_19
        # Source Nodes: [attn_output_16], Original ATen: [aten.native_dropout]
        buf84 = aten.native_dropout(reinterpret_tensor(buf83, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf90 = reinterpret_tensor(buf83, (1, 1024, 768), (786432, 768, 1), 0); del buf83  # reuse
        buf91 = buf3; del buf3  # reuse
        buf412 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18, residual_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf85, buf67, primals_109, primals_110, buf90, buf91, buf412, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_110
        buf92 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, reinterpret_tensor(buf91, (1024, 768), (768, 1), 0), primals_22, alpha=1, beta=1, out=buf92)
        del primals_21
        buf93 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf94 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, add_11, hidden_states_20, mul_10, mul_8, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf92, buf93, buf94, 3145728, grid=grid(3145728), stream=stream0)
        buf95 = reinterpret_tensor(buf22, (1024, 768), (768, 1), 0); del buf22  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_23, reinterpret_tensor(buf94, (1024, 3072), (3072, 1), 0), primals_24, alpha=1, beta=1, out=buf95)
        del primals_23
        # Source Nodes: [feed_forward_hidden_states_2], Original ATen: [aten.native_dropout]
        buf96 = aten.native_dropout(reinterpret_tensor(buf95, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf97 = buf96[0]
        buf98 = buf96[1]
        del buf96
        buf102 = reinterpret_tensor(buf95, (1, 1024, 768), (786432, 768, 1), 0); del buf95  # reuse
        buf103 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf411 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_24, residual_5, residual_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf85, buf67, buf97, primals_111, primals_112, buf102, buf103, buf411, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_112
        buf104 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, reinterpret_tensor(buf103, (1024, 768), (768, 1), 0), primals_26, alpha=1, beta=1, out=buf104)
        del primals_25
        buf105 = buf74; del buf74  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf104, (12, 64, 1024), (64, 1, 2304), 768), out=buf105)
        buf108 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_22, attn_weights_23, attn_weights_24, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_153, buf105, buf108, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_22, attn_weights_23, attn_weights_24, attn_weights_27, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf109 = aten.native_dropout(buf108, 0.1, True)
        buf110 = buf109[0]
        buf111 = buf109[1]
        del buf109
        buf112 = empty((12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf104, (12, 1024, 64), (64, 2304, 1), 1536), out=buf112)
        buf113 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf112, buf113, 786432, grid=grid(786432), stream=stream0)
        buf114 = reinterpret_tensor(buf112, (1024, 768), (768, 1), 0); del buf112  # reuse
        # Source Nodes: [x_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_27, reinterpret_tensor(buf113, (1024, 768), (768, 1), 0), primals_28, alpha=1, beta=1, out=buf114)
        del primals_27
        # Source Nodes: [attn_output_22], Original ATen: [aten.native_dropout]
        buf115 = aten.native_dropout(reinterpret_tensor(buf114, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf116 = buf115[0]
        buf117 = buf115[1]
        del buf115
        buf121 = reinterpret_tensor(buf114, (1, 1024, 768), (786432, 768, 1), 0); del buf114  # reuse
        buf122 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf410 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_26, residual_5, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf116, buf85, buf67, buf97, primals_113, primals_114, buf121, buf122, buf410, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_114
        buf123 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, reinterpret_tensor(buf122, (1024, 768), (768, 1), 0), primals_30, alpha=1, beta=1, out=buf123)
        del primals_29
        buf124 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf125 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, add_15, hidden_states_28, mul_12, mul_13, mul_14, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf123, buf124, buf125, 3145728, grid=grid(3145728), stream=stream0)
        buf126 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_31, reinterpret_tensor(buf125, (1024, 3072), (3072, 1), 0), primals_32, alpha=1, beta=1, out=buf126)
        del primals_31
        # Source Nodes: [feed_forward_hidden_states_3], Original ATen: [aten.native_dropout]
        buf127 = aten.native_dropout(reinterpret_tensor(buf126, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = buf128; del buf128  # reuse
        buf134 = reinterpret_tensor(buf126, (1, 1024, 768), (786432, 768, 1), 0); del buf126  # reuse
        buf135 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf409 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, residual_5, residual_6, residual_7, residual_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf130, buf116, buf85, buf67, buf97, primals_115, primals_116, buf134, buf135, buf409, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_116
        buf136 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, reinterpret_tensor(buf135, (1024, 768), (768, 1), 0), primals_34, alpha=1, beta=1, out=buf136)
        del primals_33
        buf137 = buf105; del buf105  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf136, (12, 64, 1024), (64, 1, 2304), 768), out=buf137)
        buf140 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_29, attn_weights_30, attn_weights_31, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_154, buf137, buf140, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_29, attn_weights_30, attn_weights_31, attn_weights_34, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf141 = aten.native_dropout(buf140, 0.1, True)
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = reinterpret_tensor(buf97, (12, 1024, 64), (65536, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf136, (12, 1024, 64), (64, 2304, 1), 1536), out=buf144)
        buf145 = reinterpret_tensor(buf85, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf85  # reuse
        # Source Nodes: [tensor_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf144, buf145, 786432, grid=grid(786432), stream=stream0)
        buf146 = reinterpret_tensor(buf144, (1024, 768), (768, 1), 0); del buf144  # reuse
        # Source Nodes: [x_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, reinterpret_tensor(buf145, (1024, 768), (768, 1), 0), primals_36, alpha=1, beta=1, out=buf146)
        del primals_35
        # Source Nodes: [attn_output_28], Original ATen: [aten.native_dropout]
        buf147 = aten.native_dropout(reinterpret_tensor(buf146, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf153 = reinterpret_tensor(buf146, (1, 1024, 768), (786432, 768, 1), 0); del buf146  # reuse
        buf154 = buf67; del buf67  # reuse
        buf408 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_34, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf148, buf130, primals_117, primals_118, buf153, buf154, buf408, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_118
        buf155 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, reinterpret_tensor(buf154, (1024, 768), (768, 1), 0), primals_38, alpha=1, beta=1, out=buf155)
        del primals_37
        buf156 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf157 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19, hidden_states_36, mul_16, mul_17, mul_18, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf155, buf156, buf157, 3145728, grid=grid(3145728), stream=stream0)
        buf158 = reinterpret_tensor(buf116, (1024, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_39, reinterpret_tensor(buf157, (1024, 3072), (3072, 1), 0), primals_40, alpha=1, beta=1, out=buf158)
        del primals_39
        # Source Nodes: [feed_forward_hidden_states_4], Original ATen: [aten.native_dropout]
        buf159 = aten.native_dropout(reinterpret_tensor(buf158, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf160 = buf159[0]
        buf161 = buf159[1]
        del buf159
        buf165 = reinterpret_tensor(buf158, (1, 1024, 768), (786432, 768, 1), 0); del buf158  # reuse
        buf166 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf407 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_40, residual_10, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf148, buf130, buf160, primals_119, primals_120, buf165, buf166, buf407, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_120
        buf167 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_41, reinterpret_tensor(buf166, (1024, 768), (768, 1), 0), primals_42, alpha=1, beta=1, out=buf167)
        del primals_41
        buf168 = buf137; del buf137  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf167, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf167, (12, 64, 1024), (64, 1, 2304), 768), out=buf168)
        buf171 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_36, attn_weights_37, attn_weights_38, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_155, buf168, buf171, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_36, attn_weights_37, attn_weights_38, attn_weights_41, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf172 = aten.native_dropout(buf171, 0.1, True)
        buf173 = buf172[0]
        buf174 = buf172[1]
        del buf172
        buf175 = empty((12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf167, (12, 1024, 64), (64, 2304, 1), 1536), out=buf175)
        buf176 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf175, buf176, 786432, grid=grid(786432), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (1024, 768), (768, 1), 0); del buf175  # reuse
        # Source Nodes: [x_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_43, reinterpret_tensor(buf176, (1024, 768), (768, 1), 0), primals_44, alpha=1, beta=1, out=buf177)
        del primals_43
        # Source Nodes: [attn_output_34], Original ATen: [aten.native_dropout]
        buf178 = aten.native_dropout(reinterpret_tensor(buf177, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf184 = reinterpret_tensor(buf177, (1, 1024, 768), (786432, 768, 1), 0); del buf177  # reuse
        buf185 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf406 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_42, residual_10, residual_11, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf179, buf148, buf130, buf160, primals_121, primals_122, buf184, buf185, buf406, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_122
        buf186 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, reinterpret_tensor(buf185, (1024, 768), (768, 1), 0), primals_46, alpha=1, beta=1, out=buf186)
        del primals_45
        buf187 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf188 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, add_23, hidden_states_44, mul_20, mul_21, mul_22, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf186, buf187, buf188, 3145728, grid=grid(3145728), stream=stream0)
        buf189 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_47, reinterpret_tensor(buf188, (1024, 3072), (3072, 1), 0), primals_48, alpha=1, beta=1, out=buf189)
        del primals_47
        # Source Nodes: [feed_forward_hidden_states_5], Original ATen: [aten.native_dropout]
        buf190 = aten.native_dropout(reinterpret_tensor(buf189, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf191 = buf190[0]
        buf192 = buf190[1]
        del buf190
        buf193 = buf191; del buf191  # reuse
        buf197 = reinterpret_tensor(buf189, (1, 1024, 768), (786432, 768, 1), 0); del buf189  # reuse
        buf198 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf405 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_48, residual_10, residual_11, residual_12, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf193, buf179, buf148, buf130, buf160, primals_123, primals_124, buf197, buf198, buf405, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_124
        buf199 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, reinterpret_tensor(buf198, (1024, 768), (768, 1), 0), primals_50, alpha=1, beta=1, out=buf199)
        del primals_49
        buf200 = buf168; del buf168  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf199, (12, 64, 1024), (64, 1, 2304), 768), out=buf200)
        buf203 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_43, attn_weights_44, attn_weights_45, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_156, buf200, buf203, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_43, attn_weights_44, attn_weights_45, attn_weights_48, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf204 = aten.native_dropout(buf203, 0.1, True)
        buf205 = buf204[0]
        buf206 = buf204[1]
        del buf204
        buf207 = reinterpret_tensor(buf179, (12, 1024, 64), (65536, 64, 1), 0); del buf179  # reuse
        # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf205, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf199, (12, 1024, 64), (64, 2304, 1), 1536), out=buf207)
        buf208 = reinterpret_tensor(buf160, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf160  # reuse
        # Source Nodes: [tensor_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf207, buf208, 786432, grid=grid(786432), stream=stream0)
        buf209 = reinterpret_tensor(buf207, (1024, 768), (768, 1), 0); del buf207  # reuse
        # Source Nodes: [x_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_51, reinterpret_tensor(buf208, (1024, 768), (768, 1), 0), primals_52, alpha=1, beta=1, out=buf209)
        del primals_51
        # Source Nodes: [attn_output_40], Original ATen: [aten.native_dropout]
        buf210 = aten.native_dropout(reinterpret_tensor(buf209, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf211 = buf210[0]
        buf212 = buf210[1]
        del buf210
        buf216 = reinterpret_tensor(buf209, (1, 1024, 768), (786432, 768, 1), 0); del buf209  # reuse
        buf217 = buf148; del buf148  # reuse
        buf404 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_50, residual_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf211, buf193, primals_125, primals_126, buf216, buf217, buf404, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_126
        buf218 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_53, reinterpret_tensor(buf217, (1024, 768), (768, 1), 0), primals_54, alpha=1, beta=1, out=buf218)
        del primals_53
        buf219 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf220 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_26, add_27, hidden_states_52, mul_24, mul_25, mul_26, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf218, buf219, buf220, 3145728, grid=grid(3145728), stream=stream0)
        buf221 = reinterpret_tensor(buf130, (1024, 768), (768, 1), 0); del buf130  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_55, reinterpret_tensor(buf220, (1024, 3072), (3072, 1), 0), primals_56, alpha=1, beta=1, out=buf221)
        del primals_55
        # Source Nodes: [feed_forward_hidden_states_6], Original ATen: [aten.native_dropout]
        buf222 = aten.native_dropout(reinterpret_tensor(buf221, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf223 = buf222[0]
        buf224 = buf222[1]
        del buf222
        buf228 = reinterpret_tensor(buf221, (1, 1024, 768), (786432, 768, 1), 0); del buf221  # reuse
        buf229 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf403 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_56, residual_13, residual_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf211, buf193, buf223, primals_127, primals_128, buf228, buf229, buf403, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_128
        buf230 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_57, reinterpret_tensor(buf229, (1024, 768), (768, 1), 0), primals_58, alpha=1, beta=1, out=buf230)
        del primals_57
        buf231 = buf200; del buf200  # reuse
        # Source Nodes: [attn_weights_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf230, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf230, (12, 64, 1024), (64, 1, 2304), 768), out=buf231)
        buf234 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_50, attn_weights_51, attn_weights_52, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_157, buf231, buf234, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_50, attn_weights_51, attn_weights_52, attn_weights_55, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf235 = aten.native_dropout(buf234, 0.1, True)
        buf236 = buf235[0]
        buf237 = buf235[1]
        del buf235
        buf238 = empty((12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf230, (12, 1024, 64), (64, 2304, 1), 1536), out=buf238)
        buf239 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf238, buf239, 786432, grid=grid(786432), stream=stream0)
        buf240 = reinterpret_tensor(buf238, (1024, 768), (768, 1), 0); del buf238  # reuse
        # Source Nodes: [x_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_59, reinterpret_tensor(buf239, (1024, 768), (768, 1), 0), primals_60, alpha=1, beta=1, out=buf240)
        del primals_59
        # Source Nodes: [attn_output_46], Original ATen: [aten.native_dropout]
        buf241 = aten.native_dropout(reinterpret_tensor(buf240, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf242 = buf241[0]
        buf243 = buf241[1]
        del buf241
        buf247 = reinterpret_tensor(buf240, (1, 1024, 768), (786432, 768, 1), 0); del buf240  # reuse
        buf248 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf402 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_58, residual_13, residual_14, residual_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf242, buf211, buf193, buf223, primals_129, primals_130, buf247, buf248, buf402, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_130
        buf249 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_61, reinterpret_tensor(buf248, (1024, 768), (768, 1), 0), primals_62, alpha=1, beta=1, out=buf249)
        del primals_61
        buf250 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf251 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_30, add_31, hidden_states_60, mul_28, mul_29, mul_30, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf249, buf250, buf251, 3145728, grid=grid(3145728), stream=stream0)
        buf252 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_63, reinterpret_tensor(buf251, (1024, 3072), (3072, 1), 0), primals_64, alpha=1, beta=1, out=buf252)
        del primals_63
        # Source Nodes: [feed_forward_hidden_states_7], Original ATen: [aten.native_dropout]
        buf253 = aten.native_dropout(reinterpret_tensor(buf252, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf254 = buf253[0]
        buf255 = buf253[1]
        del buf253
        buf256 = buf254; del buf254  # reuse
        buf260 = reinterpret_tensor(buf252, (1, 1024, 768), (786432, 768, 1), 0); del buf252  # reuse
        buf261 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf401 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_64, residual_13, residual_14, residual_15, residual_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf256, buf242, buf211, buf193, buf223, primals_131, primals_132, buf260, buf261, buf401, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_132
        buf262 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, reinterpret_tensor(buf261, (1024, 768), (768, 1), 0), primals_66, alpha=1, beta=1, out=buf262)
        del primals_65
        buf263 = buf231; del buf231  # reuse
        # Source Nodes: [attn_weights_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf262, (12, 64, 1024), (64, 1, 2304), 768), out=buf263)
        buf266 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_57, attn_weights_58, attn_weights_59, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_158, buf263, buf266, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_57, attn_weights_58, attn_weights_59, attn_weights_62, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf267 = aten.native_dropout(buf266, 0.1, True)
        buf268 = buf267[0]
        buf269 = buf267[1]
        del buf267
        buf270 = reinterpret_tensor(buf242, (12, 1024, 64), (65536, 64, 1), 0); del buf242  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf268, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf262, (12, 1024, 64), (64, 2304, 1), 1536), out=buf270)
        buf271 = reinterpret_tensor(buf223, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf223  # reuse
        # Source Nodes: [tensor_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf270, buf271, 786432, grid=grid(786432), stream=stream0)
        buf272 = reinterpret_tensor(buf270, (1024, 768), (768, 1), 0); del buf270  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_67, reinterpret_tensor(buf271, (1024, 768), (768, 1), 0), primals_68, alpha=1, beta=1, out=buf272)
        del primals_67
        # Source Nodes: [attn_output_52], Original ATen: [aten.native_dropout]
        buf273 = aten.native_dropout(reinterpret_tensor(buf272, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf274 = buf273[0]
        buf275 = buf273[1]
        del buf273
        buf279 = reinterpret_tensor(buf272, (1, 1024, 768), (786432, 768, 1), 0); del buf272  # reuse
        buf280 = buf211; del buf211  # reuse
        buf400 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66, residual_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf274, buf256, primals_133, primals_134, buf279, buf280, buf400, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_134
        buf281 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_69, reinterpret_tensor(buf280, (1024, 768), (768, 1), 0), primals_70, alpha=1, beta=1, out=buf281)
        del primals_69
        buf282 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf283 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_34, add_35, hidden_states_68, mul_32, mul_33, mul_34, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf281, buf282, buf283, 3145728, grid=grid(3145728), stream=stream0)
        buf284 = reinterpret_tensor(buf193, (1024, 768), (768, 1), 0); del buf193  # reuse
        # Source Nodes: [x_70], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_71, reinterpret_tensor(buf283, (1024, 3072), (3072, 1), 0), primals_72, alpha=1, beta=1, out=buf284)
        del primals_71
        # Source Nodes: [feed_forward_hidden_states_8], Original ATen: [aten.native_dropout]
        buf285 = aten.native_dropout(reinterpret_tensor(buf284, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf286 = buf285[0]
        buf287 = buf285[1]
        del buf285
        buf291 = reinterpret_tensor(buf284, (1, 1024, 768), (786432, 768, 1), 0); del buf284  # reuse
        buf292 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf399 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72, residual_17, residual_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf274, buf256, buf286, primals_135, primals_136, buf291, buf292, buf399, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_136
        buf293 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_73, reinterpret_tensor(buf292, (1024, 768), (768, 1), 0), primals_74, alpha=1, beta=1, out=buf293)
        del primals_73
        buf294 = buf263; del buf263  # reuse
        # Source Nodes: [attn_weights_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf293, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf293, (12, 64, 1024), (64, 1, 2304), 768), out=buf294)
        buf297 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_64, attn_weights_65, attn_weights_66, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_159, buf294, buf297, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_64, attn_weights_65, attn_weights_66, attn_weights_69, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf298 = aten.native_dropout(buf297, 0.1, True)
        buf299 = buf298[0]
        buf300 = buf298[1]
        del buf298
        buf301 = empty((12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf299, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf293, (12, 1024, 64), (64, 2304, 1), 1536), out=buf301)
        buf302 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf301, buf302, 786432, grid=grid(786432), stream=stream0)
        buf303 = reinterpret_tensor(buf301, (1024, 768), (768, 1), 0); del buf301  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_75, reinterpret_tensor(buf302, (1024, 768), (768, 1), 0), primals_76, alpha=1, beta=1, out=buf303)
        del primals_75
        # Source Nodes: [attn_output_58], Original ATen: [aten.native_dropout]
        buf304 = aten.native_dropout(reinterpret_tensor(buf303, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf305 = buf304[0]
        buf306 = buf304[1]
        del buf304
        buf310 = reinterpret_tensor(buf303, (1, 1024, 768), (786432, 768, 1), 0); del buf303  # reuse
        buf311 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf398 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_74, residual_17, residual_18, residual_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf305, buf274, buf256, buf286, primals_137, primals_138, buf310, buf311, buf398, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_138
        buf312 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, reinterpret_tensor(buf311, (1024, 768), (768, 1), 0), primals_78, alpha=1, beta=1, out=buf312)
        del primals_77
        buf313 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf314 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_38, add_39, hidden_states_76, mul_36, mul_37, mul_38, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf312, buf313, buf314, 3145728, grid=grid(3145728), stream=stream0)
        buf315 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_79, reinterpret_tensor(buf314, (1024, 3072), (3072, 1), 0), primals_80, alpha=1, beta=1, out=buf315)
        del primals_79
        # Source Nodes: [feed_forward_hidden_states_9], Original ATen: [aten.native_dropout]
        buf316 = aten.native_dropout(reinterpret_tensor(buf315, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf317 = buf316[0]
        buf318 = buf316[1]
        del buf316
        buf319 = buf317; del buf317  # reuse
        buf323 = reinterpret_tensor(buf315, (1, 1024, 768), (786432, 768, 1), 0); del buf315  # reuse
        buf324 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf397 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_80, residual_17, residual_18, residual_19, residual_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf319, buf305, buf274, buf256, buf286, primals_139, primals_140, buf323, buf324, buf397, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_140
        buf325 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, reinterpret_tensor(buf324, (1024, 768), (768, 1), 0), primals_82, alpha=1, beta=1, out=buf325)
        del primals_81
        buf326 = buf294; del buf294  # reuse
        # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf325, (12, 64, 1024), (64, 1, 2304), 768), out=buf326)
        buf329 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_71, attn_weights_72, attn_weights_73, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_160, buf326, buf329, 12288, 1024, grid=grid(12288), stream=stream0)
        # Source Nodes: [attn_weights_71, attn_weights_72, attn_weights_73, attn_weights_76, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf330 = aten.native_dropout(buf329, 0.1, True)
        buf331 = buf330[0]
        buf332 = buf330[1]
        del buf330
        buf333 = reinterpret_tensor(buf305, (12, 1024, 64), (65536, 64, 1), 0); del buf305  # reuse
        # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf331, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf325, (12, 1024, 64), (64, 2304, 1), 1536), out=buf333)
        buf334 = reinterpret_tensor(buf286, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf286  # reuse
        # Source Nodes: [tensor_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf333, buf334, 786432, grid=grid(786432), stream=stream0)
        buf335 = reinterpret_tensor(buf333, (1024, 768), (768, 1), 0); del buf333  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_83, reinterpret_tensor(buf334, (1024, 768), (768, 1), 0), primals_84, alpha=1, beta=1, out=buf335)
        del primals_83
        # Source Nodes: [attn_output_64], Original ATen: [aten.native_dropout]
        buf336 = aten.native_dropout(reinterpret_tensor(buf335, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf337 = buf336[0]
        buf338 = buf336[1]
        del buf336
        buf342 = reinterpret_tensor(buf335, (1, 1024, 768), (786432, 768, 1), 0); del buf335  # reuse
        buf343 = buf274; del buf274  # reuse
        buf396 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_82, residual_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf337, buf319, primals_141, primals_142, buf342, buf343, buf396, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_142
        buf344 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_85, reinterpret_tensor(buf343, (1024, 768), (768, 1), 0), primals_86, alpha=1, beta=1, out=buf344)
        del primals_85
        buf345 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf346 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_42, add_43, hidden_states_84, mul_40, mul_41, mul_42, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf344, buf345, buf346, 3145728, grid=grid(3145728), stream=stream0)
        buf347 = reinterpret_tensor(buf256, (1024, 768), (768, 1), 0); del buf256  # reuse
        # Source Nodes: [x_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_87, reinterpret_tensor(buf346, (1024, 3072), (3072, 1), 0), primals_88, alpha=1, beta=1, out=buf347)
        del primals_87
        # Source Nodes: [feed_forward_hidden_states_10], Original ATen: [aten.native_dropout]
        buf348 = aten.native_dropout(reinterpret_tensor(buf347, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf349 = buf348[0]
        buf350 = buf348[1]
        del buf348
        buf354 = reinterpret_tensor(buf347, (1, 1024, 768), (786432, 768, 1), 0); del buf347  # reuse
        buf355 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf395 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_88, residual_21, residual_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf337, buf319, buf349, primals_143, primals_144, buf354, buf355, buf395, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_144
        buf356 = empty((1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_89, reinterpret_tensor(buf355, (1024, 768), (768, 1), 0), primals_90, alpha=1, beta=1, out=buf356)
        del primals_89
        buf357 = buf326; del buf326  # reuse
        # Source Nodes: [attn_weights_77], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf356, (12, 1024, 64), (64, 2304, 1), 0), reinterpret_tensor(buf356, (12, 64, 1024), (64, 1, 2304), 768), out=buf357)
        buf360 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_78, attn_weights_79, attn_weights_80, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_161, buf357, buf360, 12288, 1024, grid=grid(12288), stream=stream0)
        del buf357
        # Source Nodes: [attn_weights_78, attn_weights_79, attn_weights_80, attn_weights_83, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf361 = aten.native_dropout(buf360, 0.1, True)
        buf362 = buf361[0]
        buf363 = buf361[1]
        del buf361
        buf364 = empty((12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (12, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf356, (12, 1024, 64), (64, 2304, 1), 1536), out=buf364)
        buf365 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf364, buf365, 786432, grid=grid(786432), stream=stream0)
        buf366 = reinterpret_tensor(buf364, (1024, 768), (768, 1), 0); del buf364  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_91, reinterpret_tensor(buf365, (1024, 768), (768, 1), 0), primals_92, alpha=1, beta=1, out=buf366)
        del primals_91
        # Source Nodes: [attn_output_70], Original ATen: [aten.native_dropout]
        buf367 = aten.native_dropout(reinterpret_tensor(buf366, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf368 = buf367[0]
        buf369 = buf367[1]
        del buf367
        buf373 = reinterpret_tensor(buf366, (1, 1024, 768), (786432, 768, 1), 0); del buf366  # reuse
        buf374 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf394 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_90, residual_21, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf368, buf337, buf319, buf349, primals_145, primals_146, buf373, buf374, buf394, 1024, 768, grid=grid(1024), stream=stream0)
        del primals_146
        buf375 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, reinterpret_tensor(buf374, (1024, 768), (768, 1), 0), primals_94, alpha=1, beta=1, out=buf375)
        del primals_93
        buf376 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        buf377 = empty((1, 1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_46, add_47, hidden_states_92, mul_44, mul_45, mul_46, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf375, buf376, buf377, 3145728, grid=grid(3145728), stream=stream0)
        buf378 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_95, reinterpret_tensor(buf377, (1024, 3072), (3072, 1), 0), primals_96, alpha=1, beta=1, out=buf378)
        del primals_95
        # Source Nodes: [feed_forward_hidden_states_11], Original ATen: [aten.native_dropout]
        buf379 = aten.native_dropout(reinterpret_tensor(buf378, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
        buf380 = buf379[0]
        buf381 = buf379[1]
        del buf379
        buf382 = buf380; del buf380  # reuse
        buf386 = reinterpret_tensor(buf378, (1, 1024, 768), (786432, 768, 1), 0); del buf378  # reuse
        buf387 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf393 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, hidden_states_96, l__self___transformer_ln_f, residual_21, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf382, buf368, buf337, buf319, buf349, primals_147, primals_148, buf386, buf387, buf393, 1024, 768, grid=grid(1024), stream=stream0)
        del buf319
        del buf337
        del buf349
        del buf368
        del buf382
        del primals_148
        buf388 = empty((1024, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1024, 768), (768, 1), 0), reinterpret_tensor(primals_149, (768, 2), (1, 768), 0), out=buf388)
        buf389 = empty((1, ), device='cuda', dtype=torch.int64)
        buf390 = buf389; del buf389  # reuse
        # Source Nodes: [argmax, eq, long, sub], Original ATen: [aten._to_copy, aten.argmax, aten.eq, aten.sub]
        triton_red_fused__to_copy_argmax_eq_sub_10.run(buf390, primals_162, 1, 1024, grid=grid(1), stream=stream0)
        buf391 = empty((1, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [arange_1], Original ATen: [aten.arange]
        triton_poi_fused_arange_11.run(buf391, 1, grid=grid(1), stream=stream0)
        buf392 = empty((1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [pooled_logits], Original ATen: [aten.index]
        triton_poi_fused_index_12.run(buf390, buf388, buf392, 2, grid=grid(2), stream=stream0)
        return (buf387, reinterpret_tensor(buf10, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf10, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf41, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf41, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf73, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf73, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf104, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf104, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf136, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf136, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf167, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf167, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf199, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf199, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf230, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf230, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf262, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf262, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf293, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf293, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf325, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf325, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf356, (1, 12, 1024, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf356, (1, 12, 1024, 64), (0, 64, 2304, 1), 1536), buf392, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, buf0, buf4, buf8, buf17, buf23, buf27, buf29, buf30, buf35, buf39, buf48, buf54, buf58, buf60, buf61, buf66, buf71, buf80, buf86, buf90, buf92, buf93, buf98, buf102, buf111, buf117, buf121, buf123, buf124, buf129, buf134, buf143, buf149, buf153, buf155, buf156, buf161, buf165, buf174, buf180, buf184, buf186, buf187, buf192, buf197, buf206, buf212, buf216, buf218, buf219, buf224, buf228, buf237, buf243, buf247, buf249, buf250, buf255, buf260, buf269, buf275, buf279, buf281, buf282, buf287, buf291, buf300, buf306, buf310, buf312, buf313, buf318, buf323, buf332, buf338, buf342, buf344, buf345, buf350, buf354, buf363, buf369, buf373, buf375, buf376, buf381, buf386, reinterpret_tensor(buf387, (1024, 768), (768, 1), 0), buf390, buf391, reinterpret_tensor(primals_149, (2, 768), (768, 1), 0), buf393, reinterpret_tensor(primals_96, (768, 3072), (1, 768), 0), reinterpret_tensor(buf377, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_94, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf374, (768, 1024), (1, 768), 0), buf394, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), reinterpret_tensor(buf365, (768, 1024), (1, 768), 0), reinterpret_tensor(buf362, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf356, (12, 64, 1024), (64, 1, 2304), 1536), buf360, reinterpret_tensor(buf356, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf356, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_90, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf355, (768, 1024), (1, 768), 0), buf395, reinterpret_tensor(primals_88, (768, 3072), (1, 768), 0), reinterpret_tensor(buf346, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_86, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf343, (768, 1024), (1, 768), 0), buf396, reinterpret_tensor(primals_84, (768, 768), (1, 768), 0), reinterpret_tensor(buf334, (768, 1024), (1, 768), 0), reinterpret_tensor(buf331, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf325, (12, 64, 1024), (64, 1, 2304), 1536), buf329, reinterpret_tensor(buf325, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf325, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_82, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf324, (768, 1024), (1, 768), 0), buf397, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), reinterpret_tensor(buf314, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_78, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf311, (768, 1024), (1, 768), 0), buf398, reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), reinterpret_tensor(buf302, (768, 1024), (1, 768), 0), reinterpret_tensor(buf299, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf293, (12, 64, 1024), (64, 1, 2304), 1536), buf297, reinterpret_tensor(buf293, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf293, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_74, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf292, (768, 1024), (1, 768), 0), buf399, reinterpret_tensor(primals_72, (768, 3072), (1, 768), 0), reinterpret_tensor(buf283, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_70, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf280, (768, 1024), (1, 768), 0), buf400, reinterpret_tensor(primals_68, (768, 768), (1, 768), 0), reinterpret_tensor(buf271, (768, 1024), (1, 768), 0), reinterpret_tensor(buf268, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf262, (12, 64, 1024), (64, 1, 2304), 1536), buf266, reinterpret_tensor(buf262, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf262, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_66, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf261, (768, 1024), (1, 768), 0), buf401, reinterpret_tensor(primals_64, (768, 3072), (1, 768), 0), reinterpret_tensor(buf251, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_62, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf248, (768, 1024), (1, 768), 0), buf402, reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), reinterpret_tensor(buf239, (768, 1024), (1, 768), 0), reinterpret_tensor(buf236, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf230, (12, 64, 1024), (64, 1, 2304), 1536), buf234, reinterpret_tensor(buf230, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf230, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_58, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf229, (768, 1024), (1, 768), 0), buf403, reinterpret_tensor(primals_56, (768, 3072), (1, 768), 0), reinterpret_tensor(buf220, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_54, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf217, (768, 1024), (1, 768), 0), buf404, reinterpret_tensor(primals_52, (768, 768), (1, 768), 0), reinterpret_tensor(buf208, (768, 1024), (1, 768), 0), reinterpret_tensor(buf205, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf199, (12, 64, 1024), (64, 1, 2304), 1536), buf203, reinterpret_tensor(buf199, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf199, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_50, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf198, (768, 1024), (1, 768), 0), buf405, reinterpret_tensor(primals_48, (768, 3072), (1, 768), 0), reinterpret_tensor(buf188, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_46, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf185, (768, 1024), (1, 768), 0), buf406, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), reinterpret_tensor(buf176, (768, 1024), (1, 768), 0), reinterpret_tensor(buf173, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf167, (12, 64, 1024), (64, 1, 2304), 1536), buf171, reinterpret_tensor(buf167, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf167, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_42, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf166, (768, 1024), (1, 768), 0), buf407, reinterpret_tensor(primals_40, (768, 3072), (1, 768), 0), reinterpret_tensor(buf157, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_38, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf154, (768, 1024), (1, 768), 0), buf408, reinterpret_tensor(primals_36, (768, 768), (1, 768), 0), reinterpret_tensor(buf145, (768, 1024), (1, 768), 0), reinterpret_tensor(buf142, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf136, (12, 64, 1024), (64, 1, 2304), 1536), buf140, reinterpret_tensor(buf136, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf136, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_34, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf135, (768, 1024), (1, 768), 0), buf409, reinterpret_tensor(primals_32, (768, 3072), (1, 768), 0), reinterpret_tensor(buf125, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_30, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf122, (768, 1024), (1, 768), 0), buf410, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), reinterpret_tensor(buf113, (768, 1024), (1, 768), 0), reinterpret_tensor(buf110, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf104, (12, 64, 1024), (64, 1, 2304), 1536), buf108, reinterpret_tensor(buf104, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf104, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_26, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf103, (768, 1024), (1, 768), 0), buf411, reinterpret_tensor(primals_24, (768, 3072), (1, 768), 0), reinterpret_tensor(buf94, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_22, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf91, (768, 1024), (1, 768), 0), buf412, reinterpret_tensor(primals_20, (768, 768), (1, 768), 0), reinterpret_tensor(buf82, (768, 1024), (1, 768), 0), reinterpret_tensor(buf79, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf73, (12, 64, 1024), (64, 1, 2304), 1536), buf77, reinterpret_tensor(buf73, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf73, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_18, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf72, (768, 1024), (1, 768), 0), buf413, reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), reinterpret_tensor(buf62, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_14, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf59, (768, 1024), (1, 768), 0), buf414, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), reinterpret_tensor(buf50, (768, 1024), (1, 768), 0), reinterpret_tensor(buf47, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf41, (12, 64, 1024), (64, 1, 2304), 1536), buf45, reinterpret_tensor(buf41, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf41, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_10, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf40, (768, 1024), (1, 768), 0), buf415, reinterpret_tensor(primals_8, (768, 3072), (1, 768), 0), reinterpret_tensor(buf31, (3072, 1024), (1, 3072), 0), reinterpret_tensor(primals_6, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf28, (768, 1024), (1, 768), 0), buf416, reinterpret_tensor(primals_4, (768, 768), (1, 768), 0), reinterpret_tensor(buf19, (768, 1024), (1, 768), 0), reinterpret_tensor(buf16, (12, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf10, (12, 64, 1024), (64, 1, 2304), 1536), buf14, reinterpret_tensor(buf10, (12, 64, 1024), (64, 1, 2304), 0), reinterpret_tensor(buf10, (12, 1024, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_2, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf9, (768, 1024), (1, 768), 0), buf417, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_151 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_152 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_153 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_154 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_155 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_156 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_157 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_158 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_159 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_160 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_161 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_162 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPT2ForSequenceClassification', benchmark_compiled_module)
