
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


# kernel path: /tmp/torchinductor_youkaichao/sh/cshnf4rz25ni6r4chd4cmhrj7sjixcxg5kzsokaffumfgcr62auo.py
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
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/ie/ciesnpoexe4kl7p3hcgus643sbohl7awarpya277p3ijfdsegggb.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: /tmp/torchinductor_youkaichao/ah/cahwk36cce3xni2xz4bgcznmg6vwx4n5g6dsfnt4sedjmce2zjxb.py
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lm/clmozbrttqju5tdrwmbkftuxd46knp4kfoctlsv24g4pegta2io7.py
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
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_div_full_where_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, other=0.0)
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
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp16, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3i/c3i7xn4piox74tvyxyt65qsmfkfidcm3h3nakslqawckrx3v6x6q.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (32768*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinjfeppnlg37eudutooxioq66ve7go2voxmazz3bb2vyyna4vby.py
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlnqf73s2wtzeinzljwcecekzabvjflv5qcyromaorkmug4t3iw.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/rz/crz3r5kizju4ncjgdcll73ts2e4pnvagm43mfia6ytqji67hhyvu.py
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_youkaichao/xi/cxi25pzfcdzml2dx66apem55fxzp5besc4g6yvklj4muaeu3xqnf.py
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/an/canq4j4fxoc2kisycucy3siffjnxejddpojwjsxrfjowfw6se64a.py
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcydyh4ofgua5akpmmzvlq6hor4i2o7pqqzkme5772vv6uazffi.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_6, exp_6, log, sub_19, sub_20, sum_7
triton_red_fused__log_softmax_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 511
    rnumel = 50257
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
        tmp0 = tl.load(in_ptr0 + (r1 + (50257*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (50257*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp10 = tl.load(in_ptr0 + (r1 + (50257*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.log(tmp8)
        tmp13 = tmp11 - tmp12
        tl.store(out_ptr2 + (r1 + (50257*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o2/co2xbekzvohrjzzhn3cmr2ti43htpvdcksszor5a22km22qitk3k.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_6, div_12, full_default_13, ne, neg, sum_8, sum_9, where_7
triton_per_fused_nll_loss_forward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
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
    tmp10 = tmp9 + 50257
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 50257), "index out of bounds: 0 <= tmp12 < 50257")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (50257*r0)), rmask, eviction_policy='evict_last', other=0.0)
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85 = args
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
    assert_size_stride(primals_49, (50257, 768), (768, 1))
    assert_size_stride(primals_50, (1024, 768), (768, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (50257, 768), (768, 1))
    assert_size_stride(primals_78, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_79, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_80, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_81, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_82, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_83, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_84, (1, 512), (512, 1))
    assert_size_stride(primals_85, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [position_ids_1], Original ATen: [aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_view_0.run(buf0, 512, grid=grid(512), stream=stream0)
        buf1 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding]
        triton_poi_fused_add_embedding_1.run(primals_84, primals_49, primals_50, buf1, 393216, grid=grid(393216), stream=stream0)
        del primals_49
        del primals_50
        # Source Nodes: [add, inputs_embeds, position_embeds, residual], Original ATen: [aten.add, aten.embedding, aten.native_dropout]
        buf2 = aten.native_dropout(buf1, 0.1, True)
        buf3 = buf2[0]
        buf4 = buf2[1]
        del buf2
        buf8 = buf1; del buf1  # reuse
        buf9 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf218 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_2.run(buf3, primals_51, primals_52, buf8, buf9, buf218, 512, 768, grid=grid(512), stream=stream0)
        del primals_52
        buf10 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1, reinterpret_tensor(buf9, (512, 768), (768, 1), 0), primals_2, alpha=1, beta=1, out=buf10)
        del primals_1
        buf11 = empty((12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf10, (12, 64, 512), (64, 1, 2304), 768), out=buf11)
        buf14 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_78, buf11, buf14, 6144, 512, grid=grid(6144), stream=stream0)
        # Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, attn_weights_6, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf15 = aten.native_dropout(buf14, 0.1, True)
        buf16 = buf15[0]
        buf17 = buf15[1]
        del buf15
        buf18 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf10, (12, 512, 64), (64, 2304, 1), 1536), out=buf18)
        buf19 = empty((1, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf18, buf19, 393216, grid=grid(393216), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (512, 768), (768, 1), 0); del buf18  # reuse
        # Source Nodes: [x_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_3, reinterpret_tensor(buf19, (512, 768), (768, 1), 0), primals_4, alpha=1, beta=1, out=buf20)
        del primals_3
        # Source Nodes: [attn_output_4], Original ATen: [aten.native_dropout]
        buf21 = aten.native_dropout(reinterpret_tensor(buf20, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf27 = reinterpret_tensor(buf20, (1, 512, 768), (393216, 768, 1), 0); del buf20  # reuse
        buf28 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf217 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_2, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf22, buf3, primals_53, primals_54, buf27, buf28, buf217, 512, 768, grid=grid(512), stream=stream0)
        del primals_54
        buf29 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf28, (512, 768), (768, 1), 0), primals_6, alpha=1, beta=1, out=buf29)
        del primals_5
        buf30 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf31 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf29, buf30, buf31, 1572864, grid=grid(1572864), stream=stream0)
        buf32 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, reinterpret_tensor(buf31, (512, 3072), (3072, 1), 0), primals_8, alpha=1, beta=1, out=buf32)
        del primals_7
        # Source Nodes: [feed_forward_hidden_states], Original ATen: [aten.native_dropout]
        buf33 = aten.native_dropout(reinterpret_tensor(buf32, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf39 = reinterpret_tensor(buf32, (1, 512, 768), (393216, 768, 1), 0); del buf32  # reuse
        buf40 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf216 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf22, buf3, buf34, primals_55, primals_56, buf39, buf40, buf216, 512, 768, grid=grid(512), stream=stream0)
        del primals_56
        buf41 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf40, (512, 768), (768, 1), 0), primals_10, alpha=1, beta=1, out=buf41)
        del primals_9
        buf42 = buf11; del buf11  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf41, (12, 64, 512), (64, 1, 2304), 768), out=buf42)
        buf45 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_10, attn_weights_8, attn_weights_9, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_79, buf42, buf45, 6144, 512, grid=grid(6144), stream=stream0)
        # Source Nodes: [attn_weights_10, attn_weights_13, attn_weights_8, attn_weights_9, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf46 = aten.native_dropout(buf45, 0.1, True)
        buf47 = buf46[0]
        buf48 = buf46[1]
        del buf46
        buf49 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf47, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf41, (12, 512, 64), (64, 2304, 1), 1536), out=buf49)
        buf50 = empty((1, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf49, buf50, 393216, grid=grid(393216), stream=stream0)
        buf51 = reinterpret_tensor(buf49, (512, 768), (768, 1), 0); del buf49  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, reinterpret_tensor(buf50, (512, 768), (768, 1), 0), primals_12, alpha=1, beta=1, out=buf51)
        del primals_11
        # Source Nodes: [attn_output_10], Original ATen: [aten.native_dropout]
        buf52 = aten.native_dropout(reinterpret_tensor(buf51, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf58 = reinterpret_tensor(buf51, (1, 512, 768), (393216, 768, 1), 0); del buf51  # reuse
        buf59 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf215 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_10, residual_1, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf53, buf22, buf3, buf34, primals_57, primals_58, buf58, buf59, buf215, 512, 768, grid=grid(512), stream=stream0)
        del primals_58
        buf60 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf59, (512, 768), (768, 1), 0), primals_14, alpha=1, beta=1, out=buf60)
        del primals_13
        buf61 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf62 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, add_7, hidden_states_12, mul_4, mul_5, mul_6, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf60, buf61, buf62, 1572864, grid=grid(1572864), stream=stream0)
        buf63 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_15, reinterpret_tensor(buf62, (512, 3072), (3072, 1), 0), primals_16, alpha=1, beta=1, out=buf63)
        del primals_15
        # Source Nodes: [feed_forward_hidden_states_1], Original ATen: [aten.native_dropout]
        buf64 = aten.native_dropout(reinterpret_tensor(buf63, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf65; del buf65  # reuse
        buf71 = reinterpret_tensor(buf63, (1, 512, 768), (393216, 768, 1), 0); del buf63  # reuse
        buf72 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf214 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_16, residual_1, residual_2, residual_3, residual_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf67, buf53, buf22, buf3, buf34, primals_59, primals_60, buf71, buf72, buf214, 512, 768, grid=grid(512), stream=stream0)
        del primals_60
        buf73 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, reinterpret_tensor(buf72, (512, 768), (768, 1), 0), primals_18, alpha=1, beta=1, out=buf73)
        del primals_17
        buf74 = buf42; del buf42  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf73, (12, 64, 512), (64, 1, 2304), 768), out=buf74)
        buf77 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_15, attn_weights_16, attn_weights_17, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_80, buf74, buf77, 6144, 512, grid=grid(6144), stream=stream0)
        # Source Nodes: [attn_weights_15, attn_weights_16, attn_weights_17, attn_weights_20, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf78 = aten.native_dropout(buf77, 0.1, True)
        buf79 = buf78[0]
        buf80 = buf78[1]
        del buf78
        buf81 = reinterpret_tensor(buf53, (12, 512, 64), (32768, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf73, (12, 512, 64), (64, 2304, 1), 1536), out=buf81)
        buf82 = reinterpret_tensor(buf34, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [tensor_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf81, buf82, 393216, grid=grid(393216), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (512, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, reinterpret_tensor(buf82, (512, 768), (768, 1), 0), primals_20, alpha=1, beta=1, out=buf83)
        del primals_19
        # Source Nodes: [attn_output_16], Original ATen: [aten.native_dropout]
        buf84 = aten.native_dropout(reinterpret_tensor(buf83, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf90 = reinterpret_tensor(buf83, (1, 512, 768), (393216, 768, 1), 0); del buf83  # reuse
        buf91 = buf3; del buf3  # reuse
        buf213 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18, residual_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf85, buf67, primals_61, primals_62, buf90, buf91, buf213, 512, 768, grid=grid(512), stream=stream0)
        del primals_62
        buf92 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, reinterpret_tensor(buf91, (512, 768), (768, 1), 0), primals_22, alpha=1, beta=1, out=buf92)
        del primals_21
        buf93 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf94 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, add_11, hidden_states_20, mul_10, mul_8, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf92, buf93, buf94, 1572864, grid=grid(1572864), stream=stream0)
        buf95 = reinterpret_tensor(buf22, (512, 768), (768, 1), 0); del buf22  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_23, reinterpret_tensor(buf94, (512, 3072), (3072, 1), 0), primals_24, alpha=1, beta=1, out=buf95)
        del primals_23
        # Source Nodes: [feed_forward_hidden_states_2], Original ATen: [aten.native_dropout]
        buf96 = aten.native_dropout(reinterpret_tensor(buf95, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf97 = buf96[0]
        buf98 = buf96[1]
        del buf96
        buf102 = reinterpret_tensor(buf95, (1, 512, 768), (393216, 768, 1), 0); del buf95  # reuse
        buf103 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf212 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_24, residual_5, residual_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf85, buf67, buf97, primals_63, primals_64, buf102, buf103, buf212, 512, 768, grid=grid(512), stream=stream0)
        del primals_64
        buf104 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, reinterpret_tensor(buf103, (512, 768), (768, 1), 0), primals_26, alpha=1, beta=1, out=buf104)
        del primals_25
        buf105 = buf74; del buf74  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf104, (12, 64, 512), (64, 1, 2304), 768), out=buf105)
        buf108 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_22, attn_weights_23, attn_weights_24, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_81, buf105, buf108, 6144, 512, grid=grid(6144), stream=stream0)
        # Source Nodes: [attn_weights_22, attn_weights_23, attn_weights_24, attn_weights_27, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf109 = aten.native_dropout(buf108, 0.1, True)
        buf110 = buf109[0]
        buf111 = buf109[1]
        del buf109
        buf112 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf104, (12, 512, 64), (64, 2304, 1), 1536), out=buf112)
        buf113 = empty((1, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf112, buf113, 393216, grid=grid(393216), stream=stream0)
        buf114 = reinterpret_tensor(buf112, (512, 768), (768, 1), 0); del buf112  # reuse
        # Source Nodes: [x_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_27, reinterpret_tensor(buf113, (512, 768), (768, 1), 0), primals_28, alpha=1, beta=1, out=buf114)
        del primals_27
        # Source Nodes: [attn_output_22], Original ATen: [aten.native_dropout]
        buf115 = aten.native_dropout(reinterpret_tensor(buf114, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf116 = buf115[0]
        buf117 = buf115[1]
        del buf115
        buf121 = reinterpret_tensor(buf114, (1, 512, 768), (393216, 768, 1), 0); del buf114  # reuse
        buf122 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf211 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_26, residual_5, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf116, buf85, buf67, buf97, primals_65, primals_66, buf121, buf122, buf211, 512, 768, grid=grid(512), stream=stream0)
        del primals_66
        buf123 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, reinterpret_tensor(buf122, (512, 768), (768, 1), 0), primals_30, alpha=1, beta=1, out=buf123)
        del primals_29
        buf124 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf125 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, add_15, hidden_states_28, mul_12, mul_13, mul_14, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf123, buf124, buf125, 1572864, grid=grid(1572864), stream=stream0)
        buf126 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_31, reinterpret_tensor(buf125, (512, 3072), (3072, 1), 0), primals_32, alpha=1, beta=1, out=buf126)
        del primals_31
        # Source Nodes: [feed_forward_hidden_states_3], Original ATen: [aten.native_dropout]
        buf127 = aten.native_dropout(reinterpret_tensor(buf126, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = buf128; del buf128  # reuse
        buf134 = reinterpret_tensor(buf126, (1, 512, 768), (393216, 768, 1), 0); del buf126  # reuse
        buf135 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf210 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_32, residual_5, residual_6, residual_7, residual_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf130, buf116, buf85, buf67, buf97, primals_67, primals_68, buf134, buf135, buf210, 512, 768, grid=grid(512), stream=stream0)
        del primals_68
        buf136 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, reinterpret_tensor(buf135, (512, 768), (768, 1), 0), primals_34, alpha=1, beta=1, out=buf136)
        del primals_33
        buf137 = buf105; del buf105  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf136, (12, 64, 512), (64, 1, 2304), 768), out=buf137)
        buf140 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_29, attn_weights_30, attn_weights_31, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_82, buf137, buf140, 6144, 512, grid=grid(6144), stream=stream0)
        # Source Nodes: [attn_weights_29, attn_weights_30, attn_weights_31, attn_weights_34, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf141 = aten.native_dropout(buf140, 0.1, True)
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = reinterpret_tensor(buf97, (12, 512, 64), (32768, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf136, (12, 512, 64), (64, 2304, 1), 1536), out=buf144)
        buf145 = reinterpret_tensor(buf85, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf85  # reuse
        # Source Nodes: [tensor_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf144, buf145, 393216, grid=grid(393216), stream=stream0)
        buf146 = reinterpret_tensor(buf144, (512, 768), (768, 1), 0); del buf144  # reuse
        # Source Nodes: [x_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, reinterpret_tensor(buf145, (512, 768), (768, 1), 0), primals_36, alpha=1, beta=1, out=buf146)
        del primals_35
        # Source Nodes: [attn_output_28], Original ATen: [aten.native_dropout]
        buf147 = aten.native_dropout(reinterpret_tensor(buf146, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf153 = reinterpret_tensor(buf146, (1, 512, 768), (393216, 768, 1), 0); del buf146  # reuse
        buf154 = buf67; del buf67  # reuse
        buf209 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_34, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf148, buf130, primals_69, primals_70, buf153, buf154, buf209, 512, 768, grid=grid(512), stream=stream0)
        del primals_70
        buf155 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_37, reinterpret_tensor(buf154, (512, 768), (768, 1), 0), primals_38, alpha=1, beta=1, out=buf155)
        del primals_37
        buf156 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf157 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_18, add_19, hidden_states_36, mul_16, mul_17, mul_18, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf155, buf156, buf157, 1572864, grid=grid(1572864), stream=stream0)
        buf158 = reinterpret_tensor(buf116, (512, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_39, reinterpret_tensor(buf157, (512, 3072), (3072, 1), 0), primals_40, alpha=1, beta=1, out=buf158)
        del primals_39
        # Source Nodes: [feed_forward_hidden_states_4], Original ATen: [aten.native_dropout]
        buf159 = aten.native_dropout(reinterpret_tensor(buf158, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf160 = buf159[0]
        buf161 = buf159[1]
        del buf159
        buf165 = reinterpret_tensor(buf158, (1, 512, 768), (393216, 768, 1), 0); del buf158  # reuse
        buf166 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf208 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_40, residual_10, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf148, buf130, buf160, primals_71, primals_72, buf165, buf166, buf208, 512, 768, grid=grid(512), stream=stream0)
        del primals_72
        buf167 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_41, reinterpret_tensor(buf166, (512, 768), (768, 1), 0), primals_42, alpha=1, beta=1, out=buf167)
        del primals_41
        buf168 = buf137; del buf137  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf167, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf167, (12, 64, 512), (64, 1, 2304), 768), out=buf168)
        buf171 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_36, attn_weights_37, attn_weights_38, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(primals_83, buf168, buf171, 6144, 512, grid=grid(6144), stream=stream0)
        del buf168
        # Source Nodes: [attn_weights_36, attn_weights_37, attn_weights_38, attn_weights_41, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.native_dropout, aten.where]
        buf172 = aten.native_dropout(buf171, 0.1, True)
        buf173 = buf172[0]
        buf174 = buf172[1]
        del buf172
        buf175 = empty((12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf167, (12, 512, 64), (64, 2304, 1), 1536), out=buf175)
        buf176 = empty((1, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf175, buf176, 393216, grid=grid(393216), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (512, 768), (768, 1), 0); del buf175  # reuse
        # Source Nodes: [x_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_43, reinterpret_tensor(buf176, (512, 768), (768, 1), 0), primals_44, alpha=1, beta=1, out=buf177)
        del primals_43
        # Source Nodes: [attn_output_34], Original ATen: [aten.native_dropout]
        buf178 = aten.native_dropout(reinterpret_tensor(buf177, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf184 = reinterpret_tensor(buf177, (1, 512, 768), (393216, 768, 1), 0); del buf177  # reuse
        buf185 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf207 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_42, residual_10, residual_11, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf179, buf148, buf130, buf160, primals_73, primals_74, buf184, buf185, buf207, 512, 768, grid=grid(512), stream=stream0)
        del primals_74
        buf186 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_45, reinterpret_tensor(buf185, (512, 768), (768, 1), 0), primals_46, alpha=1, beta=1, out=buf186)
        del primals_45
        buf187 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        buf188 = empty((1, 512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_22, add_23, hidden_states_44, mul_20, mul_21, mul_22, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf186, buf187, buf188, 1572864, grid=grid(1572864), stream=stream0)
        buf189 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_47, reinterpret_tensor(buf188, (512, 3072), (3072, 1), 0), primals_48, alpha=1, beta=1, out=buf189)
        del primals_47
        # Source Nodes: [feed_forward_hidden_states_5], Original ATen: [aten.native_dropout]
        buf190 = aten.native_dropout(reinterpret_tensor(buf189, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
        buf191 = buf190[0]
        buf192 = buf190[1]
        del buf190
        buf193 = buf191; del buf191  # reuse
        buf197 = reinterpret_tensor(buf189, (1, 512, 768), (393216, 768, 1), 0); del buf189  # reuse
        buf198 = empty((512, 768), device='cuda', dtype=torch.float32)
        buf206 = empty((1, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_47, hidden_states_48, l__mod___transformer_ln_f, lm_logits, residual_10, residual_11, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf193, buf179, buf148, buf130, buf160, primals_75, primals_76, buf197, buf198, buf206, 512, 768, grid=grid(512), stream=stream0)
        del buf130
        del buf148
        del buf160
        del buf179
        del buf193
        del primals_76
        buf199 = empty((512, 50257), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(buf198, reinterpret_tensor(primals_77, (768, 50257), (1, 768), 0), out=buf199)
        buf202 = empty((511, 50257), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_10.run(buf199, buf202, 511, 50257, grid=grid(511), stream=stream0)
        buf205 = empty((), device='cuda', dtype=torch.float32)
        buf204 = empty((), device='cuda', dtype=torch.float32)
        buf219 = buf205; del buf205  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_11.run(buf219, primals_85, buf202, buf204, 1, 511, grid=grid(1), stream=stream0)
        return (buf219, reinterpret_tensor(buf199, (1, 512, 50257), (25731584, 50257, 1), 0), reinterpret_tensor(buf10, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf10, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf41, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf41, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf73, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf73, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf104, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf104, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf136, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf136, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf167, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf167, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_85, primals_84, buf0, buf4, buf8, reinterpret_tensor(primals_78, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf17, buf23, buf27, buf29, buf30, buf35, buf39, reinterpret_tensor(primals_79, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf48, buf54, buf58, buf60, buf61, buf66, buf71, reinterpret_tensor(primals_80, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf80, buf86, buf90, buf92, buf93, buf98, buf102, reinterpret_tensor(primals_81, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf111, buf117, buf121, buf123, buf124, buf129, buf134, reinterpret_tensor(primals_82, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf143, buf149, buf153, buf155, buf156, buf161, buf165, reinterpret_tensor(primals_83, (1, 1, 512, 512), (1048576, 1048576, 1024, 1), 0), buf174, buf180, buf184, buf186, buf187, buf192, buf197, buf198, buf202, buf204, reinterpret_tensor(primals_77, (50257, 768), (768, 1), 0), buf206, reinterpret_tensor(primals_48, (768, 3072), (1, 768), 0), reinterpret_tensor(buf188, (3072, 512), (1, 3072), 0), reinterpret_tensor(primals_46, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf185, (768, 512), (1, 768), 0), buf207, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), reinterpret_tensor(buf176, (768, 512), (1, 768), 0), reinterpret_tensor(buf173, (12, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf167, (12, 64, 512), (64, 1, 2304), 1536), buf171, reinterpret_tensor(buf167, (12, 64, 512), (64, 1, 2304), 0), reinterpret_tensor(buf167, (12, 512, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_42, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf166, (768, 512), (1, 768), 0), buf208, reinterpret_tensor(primals_40, (768, 3072), (1, 768), 0), reinterpret_tensor(buf157, (3072, 512), (1, 3072), 0), reinterpret_tensor(primals_38, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf154, (768, 512), (1, 768), 0), buf209, reinterpret_tensor(primals_36, (768, 768), (1, 768), 0), reinterpret_tensor(buf145, (768, 512), (1, 768), 0), reinterpret_tensor(buf142, (12, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf136, (12, 64, 512), (64, 1, 2304), 1536), buf140, reinterpret_tensor(buf136, (12, 64, 512), (64, 1, 2304), 0), reinterpret_tensor(buf136, (12, 512, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_34, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf135, (768, 512), (1, 768), 0), buf210, reinterpret_tensor(primals_32, (768, 3072), (1, 768), 0), reinterpret_tensor(buf125, (3072, 512), (1, 3072), 0), reinterpret_tensor(primals_30, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf122, (768, 512), (1, 768), 0), buf211, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), reinterpret_tensor(buf113, (768, 512), (1, 768), 0), reinterpret_tensor(buf110, (12, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf104, (12, 64, 512), (64, 1, 2304), 1536), buf108, reinterpret_tensor(buf104, (12, 64, 512), (64, 1, 2304), 0), reinterpret_tensor(buf104, (12, 512, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_26, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf103, (768, 512), (1, 768), 0), buf212, reinterpret_tensor(primals_24, (768, 3072), (1, 768), 0), reinterpret_tensor(buf94, (3072, 512), (1, 3072), 0), reinterpret_tensor(primals_22, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf91, (768, 512), (1, 768), 0), buf213, reinterpret_tensor(primals_20, (768, 768), (1, 768), 0), reinterpret_tensor(buf82, (768, 512), (1, 768), 0), reinterpret_tensor(buf79, (12, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf73, (12, 64, 512), (64, 1, 2304), 1536), buf77, reinterpret_tensor(buf73, (12, 64, 512), (64, 1, 2304), 0), reinterpret_tensor(buf73, (12, 512, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_18, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf72, (768, 512), (1, 768), 0), buf214, reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), reinterpret_tensor(buf62, (3072, 512), (1, 3072), 0), reinterpret_tensor(primals_14, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf59, (768, 512), (1, 768), 0), buf215, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), reinterpret_tensor(buf50, (768, 512), (1, 768), 0), reinterpret_tensor(buf47, (12, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf41, (12, 64, 512), (64, 1, 2304), 1536), buf45, reinterpret_tensor(buf41, (12, 64, 512), (64, 1, 2304), 0), reinterpret_tensor(buf41, (12, 512, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_10, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf40, (768, 512), (1, 768), 0), buf216, reinterpret_tensor(primals_8, (768, 3072), (1, 768), 0), reinterpret_tensor(buf31, (3072, 512), (1, 3072), 0), reinterpret_tensor(primals_6, (3072, 768), (1, 3072), 0), reinterpret_tensor(buf28, (768, 512), (1, 768), 0), buf217, reinterpret_tensor(primals_4, (768, 768), (1, 768), 0), reinterpret_tensor(buf19, (768, 512), (1, 768), 0), reinterpret_tensor(buf16, (12, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf10, (12, 64, 512), (64, 1, 2304), 1536), buf14, reinterpret_tensor(buf10, (12, 64, 512), (64, 1, 2304), 0), reinterpret_tensor(buf10, (12, 512, 64), (64, 2304, 1), 768), reinterpret_tensor(primals_2, (2304, 768), (1, 2304), 0), reinterpret_tensor(buf9, (768, 512), (1, 768), 0), buf218, )


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
    primals_49 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_79 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_80 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_81 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_82 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_83 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_84 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_85 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistillGPT2', benchmark_compiled_module)
