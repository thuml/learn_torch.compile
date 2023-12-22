
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


# kernel path: /tmp/torchinductor_youkaichao/tr/ctri7r6f5im5sjsakv24pub3govasfwc5q6n7we3qqxacsuboqck.py
# Source Nodes: [add], Original ATen: [aten.add]
# add => add
triton_poi_fused_add_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex
    tmp0 = 2 + x0
    tl.store(out_ptr0 + (x2), tmp0, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/yo/cyoglffbgas7sh4yxzarq4jj2e3z7ejjuiykhrnhtse5upgkief5.py
# Source Nodes: [embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_model_encoder_embed_tokens, l__mod___model_model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# embed_pos => embedding_1
# hidden_states => add_1
# hidden_states_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub, var_mean
# inputs_embeds => mul
# l__mod___model_model_encoder_embed_tokens => embedding
# l__mod___model_model_encoder_layers_0_self_attn_q_proj => view_1
triton_red_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 512
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr2 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50265
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 50265), "index out of bounds: 0 <= tmp3 < 50265")
        tmp4 = tl.load(in_ptr1 + (r2 + (768*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr2 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp0 + 50265
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert((0 <= tmp15) & (tmp15 < 50265), "index out of bounds: 0 <= tmp15 < 50265")
        tmp16 = tl.load(in_ptr1 + (r2 + (768*tmp15)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 1.0
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp10
        tmp22 = 768.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp27, rmask)
        tl.store(out_ptr3 + (r2 + (768*x3)), tmp31, rmask)
    tmp32 = 768.0
    tmp33 = tmp11 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp36 / tmp32
    tl.store(out_ptr4 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnn3v3beix2upcyyqjcpebyypk2crwddevfz4hxzj4zn4k2jvbq.py
# Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# contiguous_2 => clone_3
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
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
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/th/cthhx643frhmf3ovaomlikbx5skyragl6r54o6oans7pz5wagygk.py
# Source Nodes: [value_states], Original ATen: [aten.clone]
# value_states => clone_2
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ov/covocirb7ajktakuvpooss4i4trl7gnsk3u2qopc3d2xhninnlll.py
# Source Nodes: [attn_weights_1], Original ATen: [aten._softmax]
# attn_weights_1 => amax, div, exp, sub_1, sum_1
triton_per_fused__softmax_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp3, 0))
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp11, rmask)
    tl.store(out_ptr0 + (x0), tmp4, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56gnygq5l3u7mranfqpngnf54g7d2k66idmotanxjsrmtqhr5gy.py
# Source Nodes: [hidden_states_3], Original ATen: [aten.view]
# hidden_states_3 => view_15
triton_poi_fused_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2q5q2t5pvuzguqjtdo2y55jjcsipnmdzwucugevcw3ju5266dl.py
# Source Nodes: [hidden_states_1, hidden_states_5, l__mod___model_model_encoder_layers_0_fc1, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_1 => add_3, mul_2
# hidden_states_5 => add_4
# l__mod___model_model_encoder_layers_0_fc1 => view_17
# residual_1 => add_5, add_6, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp28 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/zi/czi7cljhoz7ecbm56lkaefesqa3nmvswwmmvm4vrceazr3siwjnf.py
# Source Nodes: [hidden_states_7, hidden_states_9], Original ATen: [aten.gelu, aten.view]
# hidden_states_7 => add_7, erf, mul_6, mul_7, mul_8
# hidden_states_9 => view_19
triton_poi_fused_gelu_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6q/c6q2oxeue5kg7qbranxgd7xeitqbqa7otmwacfdp25grgbxx4ss2.py
# Source Nodes: [attn_probs_6, attn_weights_15], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_probs_6 => clone_53
# attn_weights_15 => amax_6, div_6, exp_6, sub_20, sum_7
triton_per_fused__softmax_clone_detach_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_8', 'mutated_arg_names': []}
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, other=0.0)
    tmp1 = r2
    tmp2 = 1 + x0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.0
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp10, 0))
    tmp12 = tmp7 - tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp13 / tmp17
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp18, rmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafuhpumkfhxhean4wvuykvwf4mtosc4tkbhhrtvmshlqqphmufw.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38605824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 50265, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 50268, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = 0.0
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3hcfdd2c4grx2oovmq3lbyzlorwfoyr2w2leafawdafhlwdziy.py
# Source Nodes: [lm_logits_1], Original ATen: [aten.add]
# lm_logits_1 => add_117
triton_poi_fused_add_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102942720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50265
    x1 = (xindex // 50265)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (50268*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264 = args
    args.clear()
    assert_size_stride(primals_1, (1026, 768), (768, 1))
    assert_size_stride(primals_2, (1026, 768), (768, 1))
    assert_size_stride(primals_3, (50265, 768), (768, 1))
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
    assert_size_stride(primals_102, (50265, 768), (768, 1))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, 768), (768, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, 768), (768, 1))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, 768), (768, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, 768), (768, 1))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, 768), (768, 1))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 768), (768, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, 768), (768, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, 768), (768, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (3072, 768), (768, 1))
    assert_size_stride(primals_126, (3072, ), (1, ))
    assert_size_stride(primals_127, (768, 3072), (3072, 1))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, 768), (768, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, 768), (768, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, 768), (768, 1))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, 768), (768, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 768), (768, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, 768), (768, 1))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 768), (768, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, 768), (768, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (3072, 768), (768, 1))
    assert_size_stride(primals_152, (3072, ), (1, ))
    assert_size_stride(primals_153, (768, 3072), (3072, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (768, ), (1, ))
    assert_size_stride(primals_157, (768, 768), (768, 1))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, 768), (768, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768), (768, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, 768), (768, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, ), (1, ))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (768, 768), (768, 1))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_169, (768, 768), (768, 1))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, 768), (768, 1))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, 768), (768, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (3072, 768), (768, 1))
    assert_size_stride(primals_178, (3072, ), (1, ))
    assert_size_stride(primals_179, (768, 3072), (3072, 1))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (768, ), (1, ))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, 768), (768, 1))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 768), (768, 1))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, 768), (768, 1))
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (768, 768), (768, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (768, 768), (768, 1))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (768, 768), (768, 1))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_197, (768, 768), (768, 1))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (768, 768), (768, 1))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (768, ), (1, ))
    assert_size_stride(primals_203, (3072, 768), (768, 1))
    assert_size_stride(primals_204, (3072, ), (1, ))
    assert_size_stride(primals_205, (768, 3072), (3072, 1))
    assert_size_stride(primals_206, (768, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (768, 768), (768, 1))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (768, 768), (768, 1))
    assert_size_stride(primals_212, (768, ), (1, ))
    assert_size_stride(primals_213, (768, 768), (768, 1))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, 768), (768, 1))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_218, (768, ), (1, ))
    assert_size_stride(primals_219, (768, 768), (768, 1))
    assert_size_stride(primals_220, (768, ), (1, ))
    assert_size_stride(primals_221, (768, 768), (768, 1))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (768, 768), (768, 1))
    assert_size_stride(primals_224, (768, ), (1, ))
    assert_size_stride(primals_225, (768, 768), (768, 1))
    assert_size_stride(primals_226, (768, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_228, (768, ), (1, ))
    assert_size_stride(primals_229, (3072, 768), (768, 1))
    assert_size_stride(primals_230, (3072, ), (1, ))
    assert_size_stride(primals_231, (768, 3072), (3072, 1))
    assert_size_stride(primals_232, (768, ), (1, ))
    assert_size_stride(primals_233, (768, ), (1, ))
    assert_size_stride(primals_234, (768, ), (1, ))
    assert_size_stride(primals_235, (768, 768), (768, 1))
    assert_size_stride(primals_236, (768, ), (1, ))
    assert_size_stride(primals_237, (768, 768), (768, 1))
    assert_size_stride(primals_238, (768, ), (1, ))
    assert_size_stride(primals_239, (768, 768), (768, 1))
    assert_size_stride(primals_240, (768, ), (1, ))
    assert_size_stride(primals_241, (768, 768), (768, 1))
    assert_size_stride(primals_242, (768, ), (1, ))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_244, (768, ), (1, ))
    assert_size_stride(primals_245, (768, 768), (768, 1))
    assert_size_stride(primals_246, (768, ), (1, ))
    assert_size_stride(primals_247, (768, 768), (768, 1))
    assert_size_stride(primals_248, (768, ), (1, ))
    assert_size_stride(primals_249, (768, 768), (768, 1))
    assert_size_stride(primals_250, (768, ), (1, ))
    assert_size_stride(primals_251, (768, 768), (768, 1))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (768, ), (1, ))
    assert_size_stride(primals_254, (768, ), (1, ))
    assert_size_stride(primals_255, (3072, 768), (768, 1))
    assert_size_stride(primals_256, (3072, ), (1, ))
    assert_size_stride(primals_257, (768, 3072), (3072, 1))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_260, (768, ), (1, ))
    assert_size_stride(primals_261, (50265, 768), (768, 1))
    assert_size_stride(primals_262, (1, 50265), (50265, 1))
    assert_size_stride(primals_263, (4, 512), (512, 1))
    assert_size_stride(primals_264, (4, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [add], Original ATen: [aten.add]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_add_0.run(buf0, 2048, grid=grid(2048), stream=stream0)
        buf4 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf501 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [embed_pos, hidden_states, hidden_states_1, inputs_embeds, l__mod___model_model_encoder_embed_tokens, l__mod___model_model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_1.run(primals_263, primals_3, primals_1, primals_4, primals_5, buf4, buf5, buf501, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_1
        del primals_3
        buf6 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf5, reinterpret_tensor(primals_6, (768, 768), (1, 768), 0), out=buf6)
        buf7 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf5, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf7)
        buf8 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf5, reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), out=buf8)
        buf9 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf6, primals_7, buf9, 1572864, grid=grid(1572864), stream=stream0)
        del primals_7
        buf10 = reinterpret_tensor(buf6, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf8, primals_11, buf10, 1572864, grid=grid(1572864), stream=stream0)
        del primals_11
        buf11 = reinterpret_tensor(buf8, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf7, primals_9, buf11, 1572864, grid=grid(1572864), stream=stream0)
        del primals_9
        buf12 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf11, (48, 64, 512), (32768, 1, 64), 0), out=buf12)
        buf13 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf14 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf15 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf12, buf13, buf14, buf15, 24576, 512, grid=grid(24576), stream=stream0)
        buf16 = reinterpret_tensor(buf7, (48, 512, 64), (32768, 64, 1), 0); del buf7  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf15, reinterpret_tensor(buf10, (48, 512, 64), (32768, 64, 1), 0), out=buf16)
        buf17 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_3], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf16, buf17, 1572864, grid=grid(1572864), stream=stream0)
        buf18 = reinterpret_tensor(buf16, (2048, 768), (768, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf17, reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), out=buf18)
        buf19 = reinterpret_tensor(buf18, (4, 512, 768), (393216, 768, 1), 0); del buf18  # reuse
        buf23 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf24 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf500 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1, hidden_states_5, l__mod___model_model_encoder_layers_0_fc1, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf19, buf4, primals_4, primals_5, primals_13, primals_14, primals_15, buf23, buf24, buf500, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_13
        del primals_5
        buf25 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_layers_0_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, buf24, reinterpret_tensor(primals_16, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del primals_17
        buf26 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_7, hidden_states_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf25, buf26, 6291456, grid=grid(6291456), stream=stream0)
        buf27 = reinterpret_tensor(buf19, (2048, 768), (768, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf26, reinterpret_tensor(primals_18, (3072, 768), (1, 3072), 0), out=buf27)
        buf28 = reinterpret_tensor(buf27, (4, 512, 768), (393216, 768, 1), 0); del buf27  # reuse
        buf32 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf33 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf499 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_11, l__mod___model_model_encoder_layers_1_self_attn_q_proj, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf28, buf23, primals_14, primals_15, primals_19, primals_20, primals_21, buf32, buf33, buf499, 2048, 768, grid=grid(2048), stream=stream0)
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
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf34, primals_23, buf37, 1572864, grid=grid(1572864), stream=stream0)
        del primals_23
        buf38 = reinterpret_tensor(buf34, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf36, primals_27, buf38, 1572864, grid=grid(1572864), stream=stream0)
        del primals_27
        buf39 = reinterpret_tensor(buf36, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf35, primals_25, buf39, 1572864, grid=grid(1572864), stream=stream0)
        del primals_25
        buf40 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf39, (48, 64, 512), (32768, 1, 64), 0), out=buf40)
        buf41 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf42 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf43 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf40, buf41, buf42, buf43, 24576, 512, grid=grid(24576), stream=stream0)
        buf44 = reinterpret_tensor(buf35, (48, 512, 64), (32768, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf43, reinterpret_tensor(buf38, (48, 512, 64), (32768, 64, 1), 0), out=buf44)
        buf45 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_14], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf44, buf45, 1572864, grid=grid(1572864), stream=stream0)
        buf46 = reinterpret_tensor(buf44, (2048, 768), (768, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf45, reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), out=buf46)
        buf47 = reinterpret_tensor(buf46, (4, 512, 768), (393216, 768, 1), 0); del buf46  # reuse
        buf51 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf52 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf498 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_16, l__mod___model_model_encoder_layers_1_fc1, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf47, buf32, primals_20, primals_21, primals_29, primals_30, primals_31, buf51, buf52, buf498, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_21
        del primals_29
        buf53 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_layers_1_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf52, reinterpret_tensor(primals_32, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf53)
        del primals_33
        buf54 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_18, hidden_states_20], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf53, buf54, 6291456, grid=grid(6291456), stream=stream0)
        buf55 = reinterpret_tensor(buf47, (2048, 768), (768, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf54, reinterpret_tensor(primals_34, (3072, 768), (1, 3072), 0), out=buf55)
        buf56 = reinterpret_tensor(buf55, (4, 512, 768), (393216, 768, 1), 0); del buf55  # reuse
        buf60 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf61 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf497 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_22, l__mod___model_model_encoder_layers_2_self_attn_q_proj, residual_3, residual_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf56, buf51, primals_30, primals_31, primals_35, primals_36, primals_37, buf60, buf61, buf497, 2048, 768, grid=grid(2048), stream=stream0)
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
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf62, primals_39, buf65, 1572864, grid=grid(1572864), stream=stream0)
        del primals_39
        buf66 = reinterpret_tensor(buf62, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf62  # reuse
        # Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf64, primals_43, buf66, 1572864, grid=grid(1572864), stream=stream0)
        del primals_43
        buf67 = reinterpret_tensor(buf64, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf64  # reuse
        # Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf63, primals_41, buf67, 1572864, grid=grid(1572864), stream=stream0)
        del primals_41
        buf68 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf67, (48, 64, 512), (32768, 1, 64), 0), out=buf68)
        buf69 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf70 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf71 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf68, buf69, buf70, buf71, 24576, 512, grid=grid(24576), stream=stream0)
        buf72 = reinterpret_tensor(buf63, (48, 512, 64), (32768, 64, 1), 0); del buf63  # reuse
        # Source Nodes: [attn_output_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf71, reinterpret_tensor(buf66, (48, 512, 64), (32768, 64, 1), 0), out=buf72)
        buf73 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_25], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf72, buf73, 1572864, grid=grid(1572864), stream=stream0)
        buf74 = reinterpret_tensor(buf72, (2048, 768), (768, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf73, reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), out=buf74)
        buf75 = reinterpret_tensor(buf74, (4, 512, 768), (393216, 768, 1), 0); del buf74  # reuse
        buf79 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf80 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf496 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_27, l__mod___model_model_encoder_layers_2_fc1, residual_4, residual_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf75, buf60, primals_36, primals_37, primals_45, primals_46, primals_47, buf79, buf80, buf496, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_37
        del primals_45
        buf81 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_layers_2_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_49, buf80, reinterpret_tensor(primals_48, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf81)
        del primals_49
        buf82 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_29, hidden_states_31], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf81, buf82, 6291456, grid=grid(6291456), stream=stream0)
        buf83 = reinterpret_tensor(buf75, (2048, 768), (768, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf82, reinterpret_tensor(primals_50, (3072, 768), (1, 3072), 0), out=buf83)
        buf84 = reinterpret_tensor(buf83, (4, 512, 768), (393216, 768, 1), 0); del buf83  # reuse
        buf88 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf89 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf495 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_33, l__mod___model_model_encoder_layers_3_self_attn_q_proj, residual_5, residual_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf84, buf79, primals_46, primals_47, primals_51, primals_52, primals_53, buf88, buf89, buf495, 2048, 768, grid=grid(2048), stream=stream0)
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
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf90, primals_55, buf93, 1572864, grid=grid(1572864), stream=stream0)
        del primals_55
        buf94 = reinterpret_tensor(buf90, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf90  # reuse
        # Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf92, primals_59, buf94, 1572864, grid=grid(1572864), stream=stream0)
        del primals_59
        buf95 = reinterpret_tensor(buf92, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf92  # reuse
        # Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf91, primals_57, buf95, 1572864, grid=grid(1572864), stream=stream0)
        del primals_57
        buf96 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf95, (48, 64, 512), (32768, 1, 64), 0), out=buf96)
        buf97 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf98 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf99 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf96, buf97, buf98, buf99, 24576, 512, grid=grid(24576), stream=stream0)
        buf100 = reinterpret_tensor(buf91, (48, 512, 64), (32768, 64, 1), 0); del buf91  # reuse
        # Source Nodes: [attn_output_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf99, reinterpret_tensor(buf94, (48, 512, 64), (32768, 64, 1), 0), out=buf100)
        buf101 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_36], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf100, buf101, 1572864, grid=grid(1572864), stream=stream0)
        buf102 = reinterpret_tensor(buf100, (2048, 768), (768, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf101, reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), out=buf102)
        buf103 = reinterpret_tensor(buf102, (4, 512, 768), (393216, 768, 1), 0); del buf102  # reuse
        buf107 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf108 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf494 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_38, l__mod___model_model_encoder_layers_3_fc1, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf103, buf88, primals_52, primals_53, primals_61, primals_62, primals_63, buf107, buf108, buf494, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_53
        del primals_61
        buf109 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_layers_3_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf108, reinterpret_tensor(primals_64, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf109)
        del primals_65
        buf110 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_40, hidden_states_42], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf109, buf110, 6291456, grid=grid(6291456), stream=stream0)
        buf111 = reinterpret_tensor(buf103, (2048, 768), (768, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf110, reinterpret_tensor(primals_66, (3072, 768), (1, 3072), 0), out=buf111)
        buf112 = reinterpret_tensor(buf111, (4, 512, 768), (393216, 768, 1), 0); del buf111  # reuse
        buf116 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf117 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf493 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_44, l__mod___model_model_encoder_layers_4_self_attn_q_proj, residual_7, residual_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf112, buf107, primals_62, primals_63, primals_67, primals_68, primals_69, buf116, buf117, buf493, 2048, 768, grid=grid(2048), stream=stream0)
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
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf118, primals_71, buf121, 1572864, grid=grid(1572864), stream=stream0)
        del primals_71
        buf122 = reinterpret_tensor(buf118, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf118  # reuse
        # Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf120, primals_75, buf122, 1572864, grid=grid(1572864), stream=stream0)
        del primals_75
        buf123 = reinterpret_tensor(buf120, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf120  # reuse
        # Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf119, primals_73, buf123, 1572864, grid=grid(1572864), stream=stream0)
        del primals_73
        buf124 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf123, (48, 64, 512), (32768, 1, 64), 0), out=buf124)
        buf125 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf126 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf127 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf124, buf125, buf126, buf127, 24576, 512, grid=grid(24576), stream=stream0)
        buf128 = reinterpret_tensor(buf119, (48, 512, 64), (32768, 64, 1), 0); del buf119  # reuse
        # Source Nodes: [attn_output_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf127, reinterpret_tensor(buf122, (48, 512, 64), (32768, 64, 1), 0), out=buf128)
        buf129 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_47], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf128, buf129, 1572864, grid=grid(1572864), stream=stream0)
        buf130 = reinterpret_tensor(buf128, (2048, 768), (768, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf129, reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), out=buf130)
        buf131 = reinterpret_tensor(buf130, (4, 512, 768), (393216, 768, 1), 0); del buf130  # reuse
        buf135 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf136 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf492 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_49, l__mod___model_model_encoder_layers_4_fc1, residual_8, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf131, buf116, primals_68, primals_69, primals_77, primals_78, primals_79, buf135, buf136, buf492, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_69
        del primals_77
        buf137 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_layers_4_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_81, buf136, reinterpret_tensor(primals_80, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf137)
        del primals_81
        buf138 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_51, hidden_states_53], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf137, buf138, 6291456, grid=grid(6291456), stream=stream0)
        buf139 = reinterpret_tensor(buf131, (2048, 768), (768, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf138, reinterpret_tensor(primals_82, (3072, 768), (1, 3072), 0), out=buf139)
        buf140 = reinterpret_tensor(buf139, (4, 512, 768), (393216, 768, 1), 0); del buf139  # reuse
        buf144 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf145 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf491 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_55, l__mod___model_model_encoder_layers_5_self_attn_q_proj, residual_10, residual_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf140, buf135, primals_78, primals_79, primals_83, primals_84, primals_85, buf144, buf145, buf491, 2048, 768, grid=grid(2048), stream=stream0)
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
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf146, primals_87, buf149, 1572864, grid=grid(1572864), stream=stream0)
        del primals_87
        buf150 = reinterpret_tensor(buf146, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf146  # reuse
        # Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf148, primals_91, buf150, 1572864, grid=grid(1572864), stream=stream0)
        del primals_91
        buf151 = reinterpret_tensor(buf148, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf148  # reuse
        # Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf147, primals_89, buf151, 1572864, grid=grid(1572864), stream=stream0)
        del primals_89
        buf152 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf151, (48, 64, 512), (32768, 1, 64), 0), out=buf152)
        buf153 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf154 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf155 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf152, buf153, buf154, buf155, 24576, 512, grid=grid(24576), stream=stream0)
        buf156 = reinterpret_tensor(buf147, (48, 512, 64), (32768, 64, 1), 0); del buf147  # reuse
        # Source Nodes: [attn_output_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf155, reinterpret_tensor(buf150, (48, 512, 64), (32768, 64, 1), 0), out=buf156)
        buf157 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_58], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf156, buf157, 1572864, grid=grid(1572864), stream=stream0)
        buf158 = reinterpret_tensor(buf156, (2048, 768), (768, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf157, reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), out=buf158)
        buf159 = reinterpret_tensor(buf158, (4, 512, 768), (393216, 768, 1), 0); del buf158  # reuse
        buf163 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf164 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf490 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_60, l__mod___model_model_encoder_layers_5_fc1, residual_10, residual_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf159, buf144, primals_84, primals_85, primals_93, primals_94, primals_95, buf163, buf164, buf490, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_85
        del primals_93
        buf165 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_layers_5_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, buf164, reinterpret_tensor(primals_96, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf165)
        del primals_97
        buf166 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_62, hidden_states_64], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf165, buf166, 6291456, grid=grid(6291456), stream=stream0)
        buf167 = reinterpret_tensor(buf159, (2048, 768), (768, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf166, reinterpret_tensor(primals_98, (3072, 768), (1, 3072), 0), out=buf167)
        buf168 = reinterpret_tensor(buf167, (4, 512, 768), (393216, 768, 1), 0); del buf167  # reuse
        buf172 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf173 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf489 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_66, hidden_states_68, residual_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf168, buf163, primals_94, primals_95, primals_99, primals_100, primals_101, buf172, buf173, buf489, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_101
        del primals_95
        del primals_99
        buf177 = buf168; del buf168  # reuse
        buf178 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf488 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_69, hidden_states_70, inputs_embeds_1, l__mod___model_model_decoder_embed_tokens, l__mod___model_model_decoder_layers_0_self_attn_q_proj, positions_2], Original ATen: [aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_embedding_mul_native_layer_norm_native_layer_norm_backward_view_1.run(primals_264, primals_102, primals_2, primals_103, primals_104, buf177, buf178, buf488, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_102
        del primals_2
        buf179 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf178, reinterpret_tensor(primals_105, (768, 768), (1, 768), 0), out=buf179)
        buf180 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf178, reinterpret_tensor(primals_107, (768, 768), (1, 768), 0), out=buf180)
        buf181 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf180, primals_108, buf181, 1572864, grid=grid(1572864), stream=stream0)
        del primals_108
        buf182 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf178, reinterpret_tensor(primals_109, (768, 768), (1, 768), 0), out=buf182)
        buf183 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf182, primals_110, buf183, 1572864, grid=grid(1572864), stream=stream0)
        del primals_110
        buf184 = reinterpret_tensor(buf182, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf182  # reuse
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf179, primals_106, buf184, 1572864, grid=grid(1572864), stream=stream0)
        del primals_106
        buf185 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf181, (48, 64, 512), (32768, 1, 64), 0), out=buf185)
        buf188 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        buf487 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_6, attn_weights_15], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_8.run(buf185, buf188, buf487, 24576, 512, grid=grid(24576), stream=stream0)
        buf189 = reinterpret_tensor(buf179, (48, 512, 64), (32768, 64, 1), 0); del buf179  # reuse
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf188, reinterpret_tensor(buf183, (48, 512, 64), (32768, 64, 1), 0), out=buf189)
        buf190 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_72], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf189, buf190, 1572864, grid=grid(1572864), stream=stream0)
        buf191 = reinterpret_tensor(buf189, (2048, 768), (768, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf190, reinterpret_tensor(primals_111, (768, 768), (1, 768), 0), out=buf191)
        buf192 = reinterpret_tensor(buf191, (4, 512, 768), (393216, 768, 1), 0); del buf191  # reuse
        buf196 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf197 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf486 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_70, hidden_states_74, l__mod___model_model_decoder_layers_0_encoder_attn_q_proj, residual_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf192, buf177, primals_103, primals_104, primals_112, primals_113, primals_114, buf196, buf197, buf486, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_104
        del primals_112
        buf198 = reinterpret_tensor(buf192, (2048, 768), (768, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf197, reinterpret_tensor(primals_115, (768, 768), (1, 768), 0), out=buf198)
        buf199 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_117, (768, 768), (1, 768), 0), out=buf199)
        buf200 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf199, primals_118, buf200, 1572864, grid=grid(1572864), stream=stream0)
        del primals_118
        buf201 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_119, (768, 768), (1, 768), 0), out=buf201)
        buf202 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf201, primals_120, buf202, 1572864, grid=grid(1572864), stream=stream0)
        del primals_120
        buf203 = reinterpret_tensor(buf201, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf201  # reuse
        # Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf198, primals_116, buf203, 1572864, grid=grid(1572864), stream=stream0)
        del primals_116
        buf204 = buf185; del buf185  # reuse
        # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf203, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf200, (48, 64, 512), (32768, 1, 64), 0), out=buf204)
        buf205 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf206 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf207 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf204, buf205, buf206, buf207, 24576, 512, grid=grid(24576), stream=stream0)
        buf208 = reinterpret_tensor(buf198, (48, 512, 64), (32768, 64, 1), 0); del buf198  # reuse
        # Source Nodes: [attn_output_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf207, reinterpret_tensor(buf202, (48, 512, 64), (32768, 64, 1), 0), out=buf208)
        buf209 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_76], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf208, buf209, 1572864, grid=grid(1572864), stream=stream0)
        buf210 = reinterpret_tensor(buf208, (2048, 768), (768, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf209, reinterpret_tensor(primals_121, (768, 768), (1, 768), 0), out=buf210)
        buf211 = reinterpret_tensor(buf210, (4, 512, 768), (393216, 768, 1), 0); del buf210  # reuse
        buf215 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf216 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf485 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_78, l__mod___model_model_decoder_layers_0_fc1, residual_13, residual_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf211, buf196, primals_113, primals_114, primals_122, primals_123, primals_124, buf215, buf216, buf485, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_114
        del primals_122
        buf217 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_layers_0_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_126, buf216, reinterpret_tensor(primals_125, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf217)
        del primals_126
        buf218 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_80, hidden_states_82], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf217, buf218, 6291456, grid=grid(6291456), stream=stream0)
        buf219 = reinterpret_tensor(buf211, (2048, 768), (768, 1), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf218, reinterpret_tensor(primals_127, (3072, 768), (1, 3072), 0), out=buf219)
        buf220 = reinterpret_tensor(buf219, (4, 512, 768), (393216, 768, 1), 0); del buf219  # reuse
        buf224 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf225 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf484 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_84, l__mod___model_model_decoder_layers_1_self_attn_q_proj, residual_14, residual_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf220, buf215, primals_123, primals_124, primals_128, primals_129, primals_130, buf224, buf225, buf484, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_124
        del primals_128
        buf226 = reinterpret_tensor(buf220, (2048, 768), (768, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf225, reinterpret_tensor(primals_131, (768, 768), (1, 768), 0), out=buf226)
        buf227 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf225, reinterpret_tensor(primals_133, (768, 768), (1, 768), 0), out=buf227)
        buf228 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf227, primals_134, buf228, 1572864, grid=grid(1572864), stream=stream0)
        del primals_134
        buf229 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf225, reinterpret_tensor(primals_135, (768, 768), (1, 768), 0), out=buf229)
        buf230 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf229, primals_136, buf230, 1572864, grid=grid(1572864), stream=stream0)
        del primals_136
        buf231 = reinterpret_tensor(buf229, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf229  # reuse
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf226, primals_132, buf231, 1572864, grid=grid(1572864), stream=stream0)
        del primals_132
        buf232 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf228, (48, 64, 512), (32768, 1, 64), 0), out=buf232)
        buf235 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        buf483 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_8, attn_weights_21], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_8.run(buf232, buf235, buf483, 24576, 512, grid=grid(24576), stream=stream0)
        buf236 = reinterpret_tensor(buf226, (48, 512, 64), (32768, 64, 1), 0); del buf226  # reuse
        # Source Nodes: [attn_output_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf235, reinterpret_tensor(buf230, (48, 512, 64), (32768, 64, 1), 0), out=buf236)
        buf237 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_87], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf236, buf237, 1572864, grid=grid(1572864), stream=stream0)
        buf238 = reinterpret_tensor(buf236, (2048, 768), (768, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf237, reinterpret_tensor(primals_137, (768, 768), (1, 768), 0), out=buf238)
        buf239 = reinterpret_tensor(buf238, (4, 512, 768), (393216, 768, 1), 0); del buf238  # reuse
        buf243 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf244 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf482 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_89, l__mod___model_model_decoder_layers_1_encoder_attn_q_proj, residual_15, residual_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf239, buf224, primals_129, primals_130, primals_138, primals_139, primals_140, buf243, buf244, buf482, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_130
        del primals_138
        buf245 = reinterpret_tensor(buf239, (2048, 768), (768, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf244, reinterpret_tensor(primals_141, (768, 768), (1, 768), 0), out=buf245)
        buf246 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_143, (768, 768), (1, 768), 0), out=buf246)
        buf247 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf246, primals_144, buf247, 1572864, grid=grid(1572864), stream=stream0)
        del primals_144
        buf248 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 768), (1, 768), 0), out=buf248)
        buf249 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf248, primals_146, buf249, 1572864, grid=grid(1572864), stream=stream0)
        del primals_146
        buf250 = reinterpret_tensor(buf248, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf248  # reuse
        # Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf245, primals_142, buf250, 1572864, grid=grid(1572864), stream=stream0)
        del primals_142
        buf251 = buf232; del buf232  # reuse
        # Source Nodes: [attn_weights_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf250, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf247, (48, 64, 512), (32768, 1, 64), 0), out=buf251)
        buf252 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf253 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf254 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf251, buf252, buf253, buf254, 24576, 512, grid=grid(24576), stream=stream0)
        buf255 = reinterpret_tensor(buf245, (48, 512, 64), (32768, 64, 1), 0); del buf245  # reuse
        # Source Nodes: [attn_output_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf254, reinterpret_tensor(buf249, (48, 512, 64), (32768, 64, 1), 0), out=buf255)
        buf256 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_91], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf255, buf256, 1572864, grid=grid(1572864), stream=stream0)
        buf257 = reinterpret_tensor(buf255, (2048, 768), (768, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf256, reinterpret_tensor(primals_147, (768, 768), (1, 768), 0), out=buf257)
        buf258 = reinterpret_tensor(buf257, (4, 512, 768), (393216, 768, 1), 0); del buf257  # reuse
        buf262 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf263 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf481 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_93, l__mod___model_model_decoder_layers_1_fc1, residual_16, residual_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf258, buf243, primals_139, primals_140, primals_148, primals_149, primals_150, buf262, buf263, buf481, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_140
        del primals_148
        buf264 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_layers_1_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf263, reinterpret_tensor(primals_151, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf264)
        del primals_152
        buf265 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_95, hidden_states_97], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf264, buf265, 6291456, grid=grid(6291456), stream=stream0)
        buf266 = reinterpret_tensor(buf258, (2048, 768), (768, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf265, reinterpret_tensor(primals_153, (3072, 768), (1, 3072), 0), out=buf266)
        buf267 = reinterpret_tensor(buf266, (4, 512, 768), (393216, 768, 1), 0); del buf266  # reuse
        buf271 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf272 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf480 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_99, l__mod___model_model_decoder_layers_2_self_attn_q_proj, residual_17, residual_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf267, buf262, primals_149, primals_150, primals_154, primals_155, primals_156, buf271, buf272, buf480, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_150
        del primals_154
        buf273 = reinterpret_tensor(buf267, (2048, 768), (768, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf272, reinterpret_tensor(primals_157, (768, 768), (1, 768), 0), out=buf273)
        buf274 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf272, reinterpret_tensor(primals_159, (768, 768), (1, 768), 0), out=buf274)
        buf275 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf274, primals_160, buf275, 1572864, grid=grid(1572864), stream=stream0)
        del primals_160
        buf276 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf272, reinterpret_tensor(primals_161, (768, 768), (1, 768), 0), out=buf276)
        buf277 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf276, primals_162, buf277, 1572864, grid=grid(1572864), stream=stream0)
        del primals_162
        buf278 = reinterpret_tensor(buf276, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf276  # reuse
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf273, primals_158, buf278, 1572864, grid=grid(1572864), stream=stream0)
        del primals_158
        buf279 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf275, (48, 64, 512), (32768, 1, 64), 0), out=buf279)
        buf282 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        buf479 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_10, attn_weights_27], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_8.run(buf279, buf282, buf479, 24576, 512, grid=grid(24576), stream=stream0)
        buf283 = reinterpret_tensor(buf273, (48, 512, 64), (32768, 64, 1), 0); del buf273  # reuse
        # Source Nodes: [attn_output_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf282, reinterpret_tensor(buf277, (48, 512, 64), (32768, 64, 1), 0), out=buf283)
        buf284 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_102], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf283, buf284, 1572864, grid=grid(1572864), stream=stream0)
        buf285 = reinterpret_tensor(buf283, (2048, 768), (768, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf284, reinterpret_tensor(primals_163, (768, 768), (1, 768), 0), out=buf285)
        buf286 = reinterpret_tensor(buf285, (4, 512, 768), (393216, 768, 1), 0); del buf285  # reuse
        buf290 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf291 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf478 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_104, l__mod___model_model_decoder_layers_2_encoder_attn_q_proj, residual_18, residual_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf286, buf271, primals_155, primals_156, primals_164, primals_165, primals_166, buf290, buf291, buf478, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_156
        del primals_164
        buf292 = reinterpret_tensor(buf286, (2048, 768), (768, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf291, reinterpret_tensor(primals_167, (768, 768), (1, 768), 0), out=buf292)
        buf293 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_169, (768, 768), (1, 768), 0), out=buf293)
        buf294 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf293, primals_170, buf294, 1572864, grid=grid(1572864), stream=stream0)
        del primals_170
        buf295 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_171, (768, 768), (1, 768), 0), out=buf295)
        buf296 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf295, primals_172, buf296, 1572864, grid=grid(1572864), stream=stream0)
        del primals_172
        buf297 = reinterpret_tensor(buf295, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf295  # reuse
        # Source Nodes: [contiguous_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf292, primals_168, buf297, 1572864, grid=grid(1572864), stream=stream0)
        del primals_168
        buf298 = buf279; del buf279  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf297, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf294, (48, 64, 512), (32768, 1, 64), 0), out=buf298)
        buf299 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf300 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf301 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_29], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf298, buf299, buf300, buf301, 24576, 512, grid=grid(24576), stream=stream0)
        buf302 = reinterpret_tensor(buf292, (48, 512, 64), (32768, 64, 1), 0); del buf292  # reuse
        # Source Nodes: [attn_output_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf301, reinterpret_tensor(buf296, (48, 512, 64), (32768, 64, 1), 0), out=buf302)
        buf303 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_106], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf302, buf303, 1572864, grid=grid(1572864), stream=stream0)
        buf304 = reinterpret_tensor(buf302, (2048, 768), (768, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf303, reinterpret_tensor(primals_173, (768, 768), (1, 768), 0), out=buf304)
        buf305 = reinterpret_tensor(buf304, (4, 512, 768), (393216, 768, 1), 0); del buf304  # reuse
        buf309 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf310 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf477 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_108, l__mod___model_model_decoder_layers_2_fc1, residual_19, residual_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf305, buf290, primals_165, primals_166, primals_174, primals_175, primals_176, buf309, buf310, buf477, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_166
        del primals_174
        buf311 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_layers_2_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_178, buf310, reinterpret_tensor(primals_177, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf311)
        del primals_178
        buf312 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_110, hidden_states_112], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf311, buf312, 6291456, grid=grid(6291456), stream=stream0)
        buf313 = reinterpret_tensor(buf305, (2048, 768), (768, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf312, reinterpret_tensor(primals_179, (3072, 768), (1, 3072), 0), out=buf313)
        buf314 = reinterpret_tensor(buf313, (4, 512, 768), (393216, 768, 1), 0); del buf313  # reuse
        buf318 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf319 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf476 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_114, l__mod___model_model_decoder_layers_3_self_attn_q_proj, residual_20, residual_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf314, buf309, primals_175, primals_176, primals_180, primals_181, primals_182, buf318, buf319, buf476, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_176
        del primals_180
        buf320 = reinterpret_tensor(buf314, (2048, 768), (768, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf319, reinterpret_tensor(primals_183, (768, 768), (1, 768), 0), out=buf320)
        buf321 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf319, reinterpret_tensor(primals_185, (768, 768), (1, 768), 0), out=buf321)
        buf322 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf321, primals_186, buf322, 1572864, grid=grid(1572864), stream=stream0)
        del primals_186
        buf323 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf319, reinterpret_tensor(primals_187, (768, 768), (1, 768), 0), out=buf323)
        buf324 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf323, primals_188, buf324, 1572864, grid=grid(1572864), stream=stream0)
        del primals_188
        buf325 = reinterpret_tensor(buf323, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf323  # reuse
        # Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf320, primals_184, buf325, 1572864, grid=grid(1572864), stream=stream0)
        del primals_184
        buf326 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf322, (48, 64, 512), (32768, 1, 64), 0), out=buf326)
        buf329 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        buf475 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_12, attn_weights_33], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_8.run(buf326, buf329, buf475, 24576, 512, grid=grid(24576), stream=stream0)
        buf330 = reinterpret_tensor(buf320, (48, 512, 64), (32768, 64, 1), 0); del buf320  # reuse
        # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf329, reinterpret_tensor(buf324, (48, 512, 64), (32768, 64, 1), 0), out=buf330)
        buf331 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_117], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf330, buf331, 1572864, grid=grid(1572864), stream=stream0)
        buf332 = reinterpret_tensor(buf330, (2048, 768), (768, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf331, reinterpret_tensor(primals_189, (768, 768), (1, 768), 0), out=buf332)
        buf333 = reinterpret_tensor(buf332, (4, 512, 768), (393216, 768, 1), 0); del buf332  # reuse
        buf337 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf338 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf474 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_119, l__mod___model_model_decoder_layers_3_encoder_attn_q_proj, residual_21, residual_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf333, buf318, primals_181, primals_182, primals_190, primals_191, primals_192, buf337, buf338, buf474, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_182
        del primals_190
        buf339 = reinterpret_tensor(buf333, (2048, 768), (768, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf338, reinterpret_tensor(primals_193, (768, 768), (1, 768), 0), out=buf339)
        buf340 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_195, (768, 768), (1, 768), 0), out=buf340)
        buf341 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf340, primals_196, buf341, 1572864, grid=grid(1572864), stream=stream0)
        del primals_196
        buf342 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_197, (768, 768), (1, 768), 0), out=buf342)
        buf343 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf342, primals_198, buf343, 1572864, grid=grid(1572864), stream=stream0)
        del primals_198
        buf344 = reinterpret_tensor(buf342, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf342  # reuse
        # Source Nodes: [contiguous_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf339, primals_194, buf344, 1572864, grid=grid(1572864), stream=stream0)
        del primals_194
        buf345 = buf326; del buf326  # reuse
        # Source Nodes: [attn_weights_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf344, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf341, (48, 64, 512), (32768, 1, 64), 0), out=buf345)
        buf346 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf347 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf348 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf345, buf346, buf347, buf348, 24576, 512, grid=grid(24576), stream=stream0)
        buf349 = reinterpret_tensor(buf339, (48, 512, 64), (32768, 64, 1), 0); del buf339  # reuse
        # Source Nodes: [attn_output_65], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf348, reinterpret_tensor(buf343, (48, 512, 64), (32768, 64, 1), 0), out=buf349)
        buf350 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_121], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf349, buf350, 1572864, grid=grid(1572864), stream=stream0)
        buf351 = reinterpret_tensor(buf349, (2048, 768), (768, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf350, reinterpret_tensor(primals_199, (768, 768), (1, 768), 0), out=buf351)
        buf352 = reinterpret_tensor(buf351, (4, 512, 768), (393216, 768, 1), 0); del buf351  # reuse
        buf356 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf357 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf473 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_123, l__mod___model_model_decoder_layers_3_fc1, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf352, buf337, primals_191, primals_192, primals_200, primals_201, primals_202, buf356, buf357, buf473, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_192
        del primals_200
        buf358 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_layers_3_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_204, buf357, reinterpret_tensor(primals_203, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf358)
        del primals_204
        buf359 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_125, hidden_states_127], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf358, buf359, 6291456, grid=grid(6291456), stream=stream0)
        buf360 = reinterpret_tensor(buf352, (2048, 768), (768, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf359, reinterpret_tensor(primals_205, (3072, 768), (1, 3072), 0), out=buf360)
        buf361 = reinterpret_tensor(buf360, (4, 512, 768), (393216, 768, 1), 0); del buf360  # reuse
        buf365 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf366 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf472 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_129, l__mod___model_model_decoder_layers_4_self_attn_q_proj, residual_23, residual_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf361, buf356, primals_201, primals_202, primals_206, primals_207, primals_208, buf365, buf366, buf472, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_202
        del primals_206
        buf367 = reinterpret_tensor(buf361, (2048, 768), (768, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf366, reinterpret_tensor(primals_209, (768, 768), (1, 768), 0), out=buf367)
        buf368 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf366, reinterpret_tensor(primals_211, (768, 768), (1, 768), 0), out=buf368)
        buf369 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf368, primals_212, buf369, 1572864, grid=grid(1572864), stream=stream0)
        del primals_212
        buf370 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf366, reinterpret_tensor(primals_213, (768, 768), (1, 768), 0), out=buf370)
        buf371 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf370, primals_214, buf371, 1572864, grid=grid(1572864), stream=stream0)
        del primals_214
        buf372 = reinterpret_tensor(buf370, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf370  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf367, primals_210, buf372, 1572864, grid=grid(1572864), stream=stream0)
        del primals_210
        buf373 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf369, (48, 64, 512), (32768, 1, 64), 0), out=buf373)
        buf376 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        buf471 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_14, attn_weights_39], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_8.run(buf373, buf376, buf471, 24576, 512, grid=grid(24576), stream=stream0)
        buf377 = reinterpret_tensor(buf367, (48, 512, 64), (32768, 64, 1), 0); del buf367  # reuse
        # Source Nodes: [attn_output_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf376, reinterpret_tensor(buf371, (48, 512, 64), (32768, 64, 1), 0), out=buf377)
        buf378 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_132], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf377, buf378, 1572864, grid=grid(1572864), stream=stream0)
        buf379 = reinterpret_tensor(buf377, (2048, 768), (768, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf378, reinterpret_tensor(primals_215, (768, 768), (1, 768), 0), out=buf379)
        buf380 = reinterpret_tensor(buf379, (4, 512, 768), (393216, 768, 1), 0); del buf379  # reuse
        buf384 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf385 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf470 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_134, l__mod___model_model_decoder_layers_4_encoder_attn_q_proj, residual_24, residual_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf380, buf365, primals_207, primals_208, primals_216, primals_217, primals_218, buf384, buf385, buf470, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_208
        del primals_216
        buf386 = reinterpret_tensor(buf380, (2048, 768), (768, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf385, reinterpret_tensor(primals_219, (768, 768), (1, 768), 0), out=buf386)
        buf387 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_221, (768, 768), (1, 768), 0), out=buf387)
        buf388 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf387, primals_222, buf388, 1572864, grid=grid(1572864), stream=stream0)
        del primals_222
        buf389 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_223, (768, 768), (1, 768), 0), out=buf389)
        buf390 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf389, primals_224, buf390, 1572864, grid=grid(1572864), stream=stream0)
        del primals_224
        buf391 = reinterpret_tensor(buf389, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf389  # reuse
        # Source Nodes: [contiguous_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf386, primals_220, buf391, 1572864, grid=grid(1572864), stream=stream0)
        del primals_220
        buf392 = buf373; del buf373  # reuse
        # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf391, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf388, (48, 64, 512), (32768, 1, 64), 0), out=buf392)
        buf393 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf394 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf395 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_41], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf392, buf393, buf394, buf395, 24576, 512, grid=grid(24576), stream=stream0)
        buf396 = reinterpret_tensor(buf386, (48, 512, 64), (32768, 64, 1), 0); del buf386  # reuse
        # Source Nodes: [attn_output_75], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf395, reinterpret_tensor(buf390, (48, 512, 64), (32768, 64, 1), 0), out=buf396)
        buf397 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_136], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf396, buf397, 1572864, grid=grid(1572864), stream=stream0)
        buf398 = reinterpret_tensor(buf396, (2048, 768), (768, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf397, reinterpret_tensor(primals_225, (768, 768), (1, 768), 0), out=buf398)
        buf399 = reinterpret_tensor(buf398, (4, 512, 768), (393216, 768, 1), 0); del buf398  # reuse
        buf403 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf404 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf469 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_138, l__mod___model_model_decoder_layers_4_fc1, residual_25, residual_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf399, buf384, primals_217, primals_218, primals_226, primals_227, primals_228, buf403, buf404, buf469, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_218
        del primals_226
        buf405 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_layers_4_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_230, buf404, reinterpret_tensor(primals_229, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf405)
        del primals_230
        buf406 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_140, hidden_states_142], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf405, buf406, 6291456, grid=grid(6291456), stream=stream0)
        buf407 = reinterpret_tensor(buf399, (2048, 768), (768, 1), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf406, reinterpret_tensor(primals_231, (3072, 768), (1, 3072), 0), out=buf407)
        buf408 = reinterpret_tensor(buf407, (4, 512, 768), (393216, 768, 1), 0); del buf407  # reuse
        buf412 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf413 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf468 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_144, l__mod___model_model_decoder_layers_5_self_attn_q_proj, residual_26, residual_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf408, buf403, primals_227, primals_228, primals_232, primals_233, primals_234, buf412, buf413, buf468, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_228
        del primals_232
        buf414 = reinterpret_tensor(buf408, (2048, 768), (768, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf413, reinterpret_tensor(primals_235, (768, 768), (1, 768), 0), out=buf414)
        buf415 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf413, reinterpret_tensor(primals_237, (768, 768), (1, 768), 0), out=buf415)
        buf416 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf415, primals_238, buf416, 1572864, grid=grid(1572864), stream=stream0)
        del primals_238
        buf417 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf413, reinterpret_tensor(primals_239, (768, 768), (1, 768), 0), out=buf417)
        buf418 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf417, primals_240, buf418, 1572864, grid=grid(1572864), stream=stream0)
        del primals_240
        buf419 = reinterpret_tensor(buf417, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf417  # reuse
        # Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf414, primals_236, buf419, 1572864, grid=grid(1572864), stream=stream0)
        del primals_236
        buf420 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf419, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf416, (48, 64, 512), (32768, 1, 64), 0), out=buf420)
        buf423 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        buf467 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs_16, attn_weights_45], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_8.run(buf420, buf423, buf467, 24576, 512, grid=grid(24576), stream=stream0)
        buf424 = reinterpret_tensor(buf414, (48, 512, 64), (32768, 64, 1), 0); del buf414  # reuse
        # Source Nodes: [attn_output_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf423, reinterpret_tensor(buf418, (48, 512, 64), (32768, 64, 1), 0), out=buf424)
        buf425 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_147], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf424, buf425, 1572864, grid=grid(1572864), stream=stream0)
        buf426 = reinterpret_tensor(buf424, (2048, 768), (768, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf425, reinterpret_tensor(primals_241, (768, 768), (1, 768), 0), out=buf426)
        buf427 = reinterpret_tensor(buf426, (4, 512, 768), (393216, 768, 1), 0); del buf426  # reuse
        buf431 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf432 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf466 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_149, l__mod___model_model_decoder_layers_5_encoder_attn_q_proj, residual_27, residual_28], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf427, buf412, primals_233, primals_234, primals_242, primals_243, primals_244, buf431, buf432, buf466, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_234
        del primals_242
        buf433 = reinterpret_tensor(buf427, (2048, 768), (768, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf432, reinterpret_tensor(primals_245, (768, 768), (1, 768), 0), out=buf433)
        buf434 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_247, (768, 768), (1, 768), 0), out=buf434)
        buf435 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf434, primals_248, buf435, 1572864, grid=grid(1572864), stream=stream0)
        del primals_248
        buf436 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_249, (768, 768), (1, 768), 0), out=buf436)
        buf437 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf436, primals_250, buf437, 1572864, grid=grid(1572864), stream=stream0)
        del primals_250
        buf438 = reinterpret_tensor(buf436, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf436  # reuse
        # Source Nodes: [contiguous_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf433, primals_246, buf438, 1572864, grid=grid(1572864), stream=stream0)
        del primals_246
        buf439 = buf420; del buf420  # reuse
        # Source Nodes: [attn_weights_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf438, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf435, (48, 64, 512), (32768, 1, 64), 0), out=buf439)
        buf440 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf441 = empty((48, 512, 1), device='cuda', dtype=torch.float32)
        buf442 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_47], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf439, buf440, buf441, buf442, 24576, 512, grid=grid(24576), stream=stream0)
        buf443 = reinterpret_tensor(buf433, (48, 512, 64), (32768, 64, 1), 0); del buf433  # reuse
        # Source Nodes: [attn_output_85], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf442, reinterpret_tensor(buf437, (48, 512, 64), (32768, 64, 1), 0), out=buf443)
        buf444 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_151], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf443, buf444, 1572864, grid=grid(1572864), stream=stream0)
        buf445 = reinterpret_tensor(buf443, (2048, 768), (768, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf444, reinterpret_tensor(primals_251, (768, 768), (1, 768), 0), out=buf445)
        buf446 = reinterpret_tensor(buf445, (4, 512, 768), (393216, 768, 1), 0); del buf445  # reuse
        buf450 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf451 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf465 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_153, l__mod___model_model_decoder_layers_5_fc1, residual_28, residual_29], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf446, buf431, primals_243, primals_244, primals_252, primals_253, primals_254, buf450, buf451, buf465, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_244
        del primals_252
        buf452 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_layers_5_fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_256, buf451, reinterpret_tensor(primals_255, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf452)
        del primals_256
        buf453 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_155, hidden_states_157], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf452, buf453, 6291456, grid=grid(6291456), stream=stream0)
        buf454 = reinterpret_tensor(buf446, (2048, 768), (768, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf453, reinterpret_tensor(primals_257, (3072, 768), (1, 3072), 0), out=buf454)
        buf455 = reinterpret_tensor(buf454, (4, 512, 768), (393216, 768, 1), 0); del buf454  # reuse
        buf459 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        buf460 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf464 = empty((4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_159, hidden_states_161, lm_logits, residual_29], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf455, buf450, primals_253, primals_254, primals_258, primals_259, primals_260, buf459, buf460, buf464, 2048, 768, grid=grid(2048), stream=stream0)
        del buf455
        del primals_254
        del primals_258
        del primals_260
        buf461 = empty_strided((768, 50268), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(primals_261, buf461, 38605824, grid=grid(38605824), stream=stream0)
        buf462 = empty((2048, 50268), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf460, buf461, out=buf462)
        del buf461
        buf463 = empty((4, 512, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits_1], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf462, primals_262, buf463, 102942720, grid=grid(102942720), stream=stream0)
        del buf462
        del primals_262
        return (buf463, buf181, buf183, buf200, buf202, buf228, buf230, buf247, buf249, buf275, buf277, buf294, buf296, buf322, buf324, buf341, buf343, buf369, buf371, buf388, buf390, buf416, buf418, buf435, buf437, buf173, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_103, primals_113, primals_123, primals_129, primals_139, primals_149, primals_155, primals_165, primals_175, primals_181, primals_191, primals_201, primals_207, primals_217, primals_227, primals_233, primals_243, primals_253, primals_259, primals_264, primals_263, buf0, buf4, buf5, buf12, buf13, buf14, buf17, buf23, buf24, buf25, buf26, buf32, buf33, buf40, buf41, buf42, buf45, buf51, buf52, buf53, buf54, buf60, buf61, buf68, buf69, buf70, buf73, buf79, buf80, buf81, buf82, buf88, buf89, buf96, buf97, buf98, buf101, buf107, buf108, buf109, buf110, buf116, buf117, buf124, buf125, buf126, buf129, buf135, buf136, buf137, buf138, buf144, buf145, buf152, buf153, buf154, buf157, buf163, buf164, buf165, buf166, buf172, buf177, buf178, buf190, buf196, buf197, reinterpret_tensor(buf173, (2048, 768), (768, 1), 0), buf204, buf205, buf206, buf209, buf215, buf216, buf217, buf218, buf224, buf225, buf237, buf243, buf244, buf251, buf252, buf253, buf256, buf262, buf263, buf264, buf265, buf271, buf272, buf284, buf290, buf291, buf298, buf299, buf300, buf303, buf309, buf310, buf311, buf312, buf318, buf319, buf331, buf337, buf338, buf345, buf346, buf347, buf350, buf356, buf357, buf358, buf359, buf365, buf366, buf378, buf384, buf385, buf392, buf393, buf394, buf397, buf403, buf404, buf405, buf406, buf412, buf413, buf425, buf431, buf432, buf439, buf440, buf441, buf444, buf450, buf451, buf452, buf453, buf459, buf460, reinterpret_tensor(primals_261, (50265, 768), (768, 1), 0), buf464, reinterpret_tensor(primals_257, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_255, (3072, 768), (768, 1), 0), buf465, reinterpret_tensor(primals_251, (768, 768), (768, 1), 0), reinterpret_tensor(buf442, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf437, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf438, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf435, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_249, (768, 768), (768, 1), 0), reinterpret_tensor(primals_247, (768, 768), (768, 1), 0), reinterpret_tensor(primals_245, (768, 768), (768, 1), 0), buf466, reinterpret_tensor(primals_241, (768, 768), (768, 1), 0), reinterpret_tensor(buf423, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf418, (48, 64, 512), (32768, 1, 64), 0), buf467, reinterpret_tensor(buf419, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf416, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_239, (768, 768), (768, 1), 0), reinterpret_tensor(primals_237, (768, 768), (768, 1), 0), reinterpret_tensor(primals_235, (768, 768), (768, 1), 0), buf468, reinterpret_tensor(primals_231, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_229, (3072, 768), (768, 1), 0), buf469, reinterpret_tensor(primals_225, (768, 768), (768, 1), 0), reinterpret_tensor(buf395, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf390, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf391, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf388, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_223, (768, 768), (768, 1), 0), reinterpret_tensor(primals_221, (768, 768), (768, 1), 0), reinterpret_tensor(primals_219, (768, 768), (768, 1), 0), buf470, reinterpret_tensor(primals_215, (768, 768), (768, 1), 0), reinterpret_tensor(buf376, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf371, (48, 64, 512), (32768, 1, 64), 0), buf471, reinterpret_tensor(buf372, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf369, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_213, (768, 768), (768, 1), 0), reinterpret_tensor(primals_211, (768, 768), (768, 1), 0), reinterpret_tensor(primals_209, (768, 768), (768, 1), 0), buf472, reinterpret_tensor(primals_205, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_203, (3072, 768), (768, 1), 0), buf473, reinterpret_tensor(primals_199, (768, 768), (768, 1), 0), reinterpret_tensor(buf348, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf343, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf344, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf341, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_197, (768, 768), (768, 1), 0), reinterpret_tensor(primals_195, (768, 768), (768, 1), 0), reinterpret_tensor(primals_193, (768, 768), (768, 1), 0), buf474, reinterpret_tensor(primals_189, (768, 768), (768, 1), 0), reinterpret_tensor(buf329, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf324, (48, 64, 512), (32768, 1, 64), 0), buf475, reinterpret_tensor(buf325, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf322, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_187, (768, 768), (768, 1), 0), reinterpret_tensor(primals_185, (768, 768), (768, 1), 0), reinterpret_tensor(primals_183, (768, 768), (768, 1), 0), buf476, reinterpret_tensor(primals_179, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_177, (3072, 768), (768, 1), 0), buf477, reinterpret_tensor(primals_173, (768, 768), (768, 1), 0), reinterpret_tensor(buf301, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf296, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf297, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf294, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_171, (768, 768), (768, 1), 0), reinterpret_tensor(primals_169, (768, 768), (768, 1), 0), reinterpret_tensor(primals_167, (768, 768), (768, 1), 0), buf478, reinterpret_tensor(primals_163, (768, 768), (768, 1), 0), reinterpret_tensor(buf282, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf277, (48, 64, 512), (32768, 1, 64), 0), buf479, reinterpret_tensor(buf278, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf275, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_161, (768, 768), (768, 1), 0), reinterpret_tensor(primals_159, (768, 768), (768, 1), 0), reinterpret_tensor(primals_157, (768, 768), (768, 1), 0), buf480, reinterpret_tensor(primals_153, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_151, (3072, 768), (768, 1), 0), buf481, reinterpret_tensor(primals_147, (768, 768), (768, 1), 0), reinterpret_tensor(buf254, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf249, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf250, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf247, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_145, (768, 768), (768, 1), 0), reinterpret_tensor(primals_143, (768, 768), (768, 1), 0), reinterpret_tensor(primals_141, (768, 768), (768, 1), 0), buf482, reinterpret_tensor(primals_137, (768, 768), (768, 1), 0), reinterpret_tensor(buf235, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf230, (48, 64, 512), (32768, 1, 64), 0), buf483, reinterpret_tensor(buf231, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf228, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_135, (768, 768), (768, 1), 0), reinterpret_tensor(primals_133, (768, 768), (768, 1), 0), reinterpret_tensor(primals_131, (768, 768), (768, 1), 0), buf484, reinterpret_tensor(primals_127, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_125, (3072, 768), (768, 1), 0), buf485, reinterpret_tensor(primals_121, (768, 768), (768, 1), 0), reinterpret_tensor(buf207, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf202, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf203, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf200, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_119, (768, 768), (768, 1), 0), reinterpret_tensor(primals_117, (768, 768), (768, 1), 0), reinterpret_tensor(primals_115, (768, 768), (768, 1), 0), buf486, reinterpret_tensor(primals_111, (768, 768), (768, 1), 0), reinterpret_tensor(buf188, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf183, (48, 64, 512), (32768, 1, 64), 0), buf487, reinterpret_tensor(buf184, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf181, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_109, (768, 768), (768, 1), 0), reinterpret_tensor(primals_107, (768, 768), (768, 1), 0), reinterpret_tensor(primals_105, (768, 768), (768, 1), 0), buf488, buf489, reinterpret_tensor(primals_98, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_96, (3072, 768), (768, 1), 0), buf490, reinterpret_tensor(primals_92, (768, 768), (768, 1), 0), reinterpret_tensor(buf155, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf150, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf149, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf151, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_90, (768, 768), (768, 1), 0), reinterpret_tensor(primals_88, (768, 768), (768, 1), 0), reinterpret_tensor(primals_86, (768, 768), (768, 1), 0), buf491, reinterpret_tensor(primals_82, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 768), (768, 1), 0), buf492, reinterpret_tensor(primals_76, (768, 768), (768, 1), 0), reinterpret_tensor(buf127, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf122, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf121, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf123, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_74, (768, 768), (768, 1), 0), reinterpret_tensor(primals_72, (768, 768), (768, 1), 0), reinterpret_tensor(primals_70, (768, 768), (768, 1), 0), buf493, reinterpret_tensor(primals_66, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_64, (3072, 768), (768, 1), 0), buf494, reinterpret_tensor(primals_60, (768, 768), (768, 1), 0), reinterpret_tensor(buf99, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf94, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf93, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf95, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_58, (768, 768), (768, 1), 0), reinterpret_tensor(primals_56, (768, 768), (768, 1), 0), reinterpret_tensor(primals_54, (768, 768), (768, 1), 0), buf495, reinterpret_tensor(primals_50, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_48, (3072, 768), (768, 1), 0), buf496, reinterpret_tensor(primals_44, (768, 768), (768, 1), 0), reinterpret_tensor(buf71, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf66, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf65, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf67, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_42, (768, 768), (768, 1), 0), reinterpret_tensor(primals_40, (768, 768), (768, 1), 0), reinterpret_tensor(primals_38, (768, 768), (768, 1), 0), buf497, reinterpret_tensor(primals_34, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_32, (3072, 768), (768, 1), 0), buf498, reinterpret_tensor(primals_28, (768, 768), (768, 1), 0), reinterpret_tensor(buf43, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf38, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf37, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf39, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_26, (768, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 768), (768, 1), 0), buf499, reinterpret_tensor(primals_18, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_16, (3072, 768), (768, 1), 0), buf500, reinterpret_tensor(primals_12, (768, 768), (768, 1), 0), reinterpret_tensor(buf15, (48, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf10, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf9, (48, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf11, (48, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 768), (768, 1), 0), reinterpret_tensor(primals_6, (768, 768), (768, 1), 0), buf501, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1026, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1026, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
    primals_102 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_264 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Bart', benchmark_compiled_module)
