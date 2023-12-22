
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


# kernel path: /tmp/torchinductor_youkaichao/ni/cniy7ooanyonddg3aamffw7drzob7zr3aaj5fj6a6f72fgbhf5o3.py
# Source Nodes: [add, hidden_states, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# add => add
# hidden_states => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# inputs_embeds => embedding
# position_embeds => embedding_1
triton_red_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50257
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50257")
        tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 + 50257
        tmp12 = tmp0 < 0
        tmp13 = tl.where(tmp12, tmp11, tmp0)
        tl.device_assert(((0 <= tmp13) & (tmp13 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp13 < 50257")
        tmp14 = tl.load(in_ptr1 + (r1 + (768*tmp13)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = 768.0
        tmp19 = tmp9 / tmp18
        tmp20 = 1e-05
        tmp21 = tmp19 + tmp20
        tmp22 = tl.math.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/2g/c2g52kzad3dtlluaz6hecttp7x4kq2r5v2kh3p2mwaqtkdlotrb4.py
# Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
# attn_weights_1 => div
# attn_weights_2 => where
# attn_weights_3 => amax, div_1, exp, sub_1, sum_1
# full => full_default
# mask_value => full_default_1
triton_per_fused__softmax__to_copy_div_full_where_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_div_full_where_1', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hp/chpkoq4xazltpbufrolfjpautcrim3ef5xbtehcacotkih4itzdx.py
# Source Nodes: [tensor_3], Original ATen: [aten.clone]
# tensor_3 => clone_2
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vd/cvdvgccyiumatnbzg3whkjzvs6pbmqamwla5xnadaitzbk2q7si4.py
# Source Nodes: [add, hidden_states_2, inputs_embeds, position_embeds, residual_1], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# add => add
# hidden_states_2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# inputs_embeds => embedding
# position_embeds => embedding_1
# residual_1 => add_3
triton_per_fused_add_embedding_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + 50257
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp6 < 50257")
    tmp7 = tl.load(in_ptr2 + (r1 + (768*tmp6)), rmask & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgpwfjx2bxdtqpf5dv73on7e2k3x7v5xgrpdek2seq7pmxwetbh.py
# Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_2 => add_6
# add_3 => add_7
# hidden_states_4 => mul_7
# mul => mul_4
# mul_1 => mul_5
# mul_2 => mul_6
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/5n/c5nci2vzvumojmswrjrkqd35pg72kilcqtq6dw422czrjo6nymlo.py
# Source Nodes: [hidden_states_8, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_8 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
# residual_2 => add_8
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvubz63a4nb7pah6wuevh3lspqyk47kqzlfxjzokilgj47ervy6.py
# Source Nodes: [hidden_states_10, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_10 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
# residual_2 => add_8
# residual_3 => add_11
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tmp2 + tmp7
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
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g3/cg3vwvse67c7wk3aqixznf77xqtrmspqosp7gwvtmggplwlyrmwj.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_6, exp_6, sub_19, sum_7
triton_red_fused__log_softmax_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (50257*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tl/ctlwigmkkougtl5bnewuz6264gqh3qkql566wbbzcyrtdpuy4dbu.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_6, div_12, full_default_13, ne_1, ne_2, neg, sum_8, sum_9, where_7
triton_per_fused_nll_loss_forward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp5 = tmp4 + 50257
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 50257), "index out of bounds: 0 <= tmp7 < 50257")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (50257*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2304, ), (1, ))
    assert_size_stride(arg1_1, (768, 2304), (2304, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, 768), (768, 1))
    assert_size_stride(arg4_1, (3072, ), (1, ))
    assert_size_stride(arg5_1, (768, 3072), (3072, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (3072, 768), (768, 1))
    assert_size_stride(arg8_1, (2304, ), (1, ))
    assert_size_stride(arg9_1, (768, 2304), (2304, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (3072, ), (1, ))
    assert_size_stride(arg13_1, (768, 3072), (3072, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, 768), (768, 1))
    assert_size_stride(arg16_1, (2304, ), (1, ))
    assert_size_stride(arg17_1, (768, 2304), (2304, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, 768), (768, 1))
    assert_size_stride(arg20_1, (3072, ), (1, ))
    assert_size_stride(arg21_1, (768, 3072), (3072, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (3072, 768), (768, 1))
    assert_size_stride(arg24_1, (2304, ), (1, ))
    assert_size_stride(arg25_1, (768, 2304), (2304, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (3072, ), (1, ))
    assert_size_stride(arg29_1, (768, 3072), (3072, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (2304, ), (1, ))
    assert_size_stride(arg33_1, (768, 2304), (2304, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, 768), (768, 1))
    assert_size_stride(arg36_1, (3072, ), (1, ))
    assert_size_stride(arg37_1, (768, 3072), (3072, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (3072, 768), (768, 1))
    assert_size_stride(arg40_1, (2304, ), (1, ))
    assert_size_stride(arg41_1, (768, 2304), (2304, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (3072, ), (1, ))
    assert_size_stride(arg45_1, (768, 3072), (3072, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (50257, 768), (768, 1))
    assert_size_stride(arg49_1, (1024, 768), (768, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (50257, 768), (768, 1))
    assert_size_stride(arg77_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg78_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg79_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg80_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg81_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg82_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg83_1, (1, 512), (512, 1))
    assert_size_stride(arg84_1, (1, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, hidden_states, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_native_layer_norm_0.run(arg83_1, arg48_1, arg49_1, arg50_1, arg51_1, buf3, 512, 768, grid=grid(512), stream=stream0)
        del arg50_1
        del arg51_1
        buf4 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg0_1, reinterpret_tensor(buf3, (512, 768), (768, 1), 0), arg1_1, alpha=1, beta=1, out=buf4)
        del arg0_1
        del arg1_1
        buf5 = empty((12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf4, (12, 64, 512), (64, 1, 2304), 768), out=buf5)
        buf8 = empty((1, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_1.run(arg77_1, buf5, buf8, 6144, 512, grid=grid(6144), stream=stream0)
        del arg77_1
        buf9 = reinterpret_tensor(buf3, (12, 512, 64), (32768, 64, 1), 0); del buf3  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf4, (12, 512, 64), (64, 2304, 1), 1536), out=buf9)
        buf10 = empty((1, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf9, buf10, 393216, grid=grid(393216), stream=stream0)
        buf11 = reinterpret_tensor(buf9, (512, 768), (768, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf10, (512, 768), (768, 1), 0), arg3_1, out=buf11)
        del arg3_1
        buf12 = reinterpret_tensor(buf11, (1, 512, 768), (393216, 768, 1), 0); del buf11  # reuse
        buf16 = reinterpret_tensor(buf10, (1, 512, 768), (393216, 768, 1), 0); del buf10  # reuse
        # Source Nodes: [add, hidden_states_2, inputs_embeds, position_embeds, residual_1], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        triton_per_fused_add_embedding_native_layer_norm_3.run(buf12, arg2_1, arg83_1, arg48_1, arg49_1, arg52_1, arg53_1, buf16, 512, 768, grid=grid(512), stream=stream0)
        del arg2_1
        del arg48_1
        del arg49_1
        del arg52_1
        del arg53_1
        del arg83_1
        buf17 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf16, (512, 768), (768, 1), 0), arg5_1, out=buf17)
        del arg5_1
        buf18 = reinterpret_tensor(buf17, (1, 512, 3072), (1572864, 3072, 1), 0); del buf17  # reuse
        # Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf18, arg4_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg4_1
        buf19 = reinterpret_tensor(buf16, (512, 768), (768, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0), arg7_1, out=buf19)
        del arg7_1
        buf23 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf12, buf19, arg6_1, arg54_1, arg55_1, buf23, 512, 768, grid=grid(512), stream=stream0)
        del arg54_1
        del arg55_1
        buf24 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf23, (512, 768), (768, 1), 0), arg9_1, alpha=1, beta=1, out=buf24)
        del arg8_1
        del arg9_1
        buf25 = reinterpret_tensor(buf8, (12, 512, 512), (262144, 512, 1), 0); del buf8  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf24, (12, 64, 512), (64, 1, 2304), 768), out=buf25)
        buf28 = reinterpret_tensor(buf5, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf5  # reuse
        # Source Nodes: [attn_weights_10, attn_weights_8, attn_weights_9, full_2, mask_value_1], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_1.run(arg78_1, buf25, buf28, 6144, 512, grid=grid(6144), stream=stream0)
        del arg78_1
        buf29 = reinterpret_tensor(buf23, (12, 512, 64), (32768, 64, 1), 0); del buf23  # reuse
        # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf24, (12, 512, 64), (64, 2304, 1), 1536), out=buf29)
        buf30 = empty((1, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [tensor_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf29, buf30, 393216, grid=grid(393216), stream=stream0)
        buf31 = reinterpret_tensor(buf29, (512, 768), (768, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (512, 768), (768, 1), 0), arg11_1, out=buf31)
        del arg11_1
        buf32 = reinterpret_tensor(buf31, (1, 512, 768), (393216, 768, 1), 0); del buf31  # reuse
        buf36 = reinterpret_tensor(buf30, (1, 512, 768), (393216, 768, 1), 0); del buf30  # reuse
        # Source Nodes: [hidden_states_10, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf32, arg10_1, buf12, buf19, arg6_1, arg56_1, arg57_1, buf36, 512, 768, grid=grid(512), stream=stream0)
        del arg10_1
        del arg56_1
        del arg57_1
        del arg6_1
        buf37 = reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (512, 768), (768, 1), 0), arg13_1, out=buf37)
        del arg13_1
        buf38 = reinterpret_tensor(buf37, (1, 512, 3072), (1572864, 3072, 1), 0); del buf37  # reuse
        # Source Nodes: [add_6, add_7, hidden_states_12, mul_4, mul_5, mul_6, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf38, arg12_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg12_1
        buf39 = reinterpret_tensor(buf36, (512, 768), (768, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (512, 3072), (3072, 1), 0), arg15_1, out=buf39)
        del arg15_1
        buf43 = reinterpret_tensor(buf19, (1, 512, 768), (393216, 768, 1), 0); del buf19  # reuse
        # Source Nodes: [hidden_states_16, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf32, buf39, arg14_1, arg58_1, arg59_1, buf43, 512, 768, grid=grid(512), stream=stream0)
        del arg58_1
        del arg59_1
        buf44 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg16_1, reinterpret_tensor(buf43, (512, 768), (768, 1), 0), arg17_1, alpha=1, beta=1, out=buf44)
        del arg16_1
        del arg17_1
        buf45 = reinterpret_tensor(buf28, (12, 512, 512), (262144, 512, 1), 0); del buf28  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf44, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf44, (12, 64, 512), (64, 1, 2304), 768), out=buf45)
        buf48 = reinterpret_tensor(buf25, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf25  # reuse
        # Source Nodes: [attn_weights_15, attn_weights_16, attn_weights_17, full_4, mask_value_2], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_1.run(arg79_1, buf45, buf48, 6144, 512, grid=grid(6144), stream=stream0)
        del arg79_1
        buf49 = reinterpret_tensor(buf43, (12, 512, 64), (32768, 64, 1), 0); del buf43  # reuse
        # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf44, (12, 512, 64), (64, 2304, 1), 1536), out=buf49)
        buf50 = reinterpret_tensor(buf12, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf12  # reuse
        # Source Nodes: [tensor_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf49, buf50, 393216, grid=grid(393216), stream=stream0)
        buf51 = reinterpret_tensor(buf49, (512, 768), (768, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (512, 768), (768, 1), 0), arg19_1, out=buf51)
        del arg19_1
        buf52 = reinterpret_tensor(buf51, (1, 512, 768), (393216, 768, 1), 0); del buf51  # reuse
        buf56 = reinterpret_tensor(buf50, (1, 512, 768), (393216, 768, 1), 0); del buf50  # reuse
        # Source Nodes: [hidden_states_18, residual_4, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf52, arg18_1, buf32, buf39, arg14_1, arg60_1, arg61_1, buf56, 512, 768, grid=grid(512), stream=stream0)
        del arg14_1
        del arg18_1
        del arg60_1
        del arg61_1
        buf57 = reinterpret_tensor(buf38, (512, 3072), (3072, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (512, 768), (768, 1), 0), arg21_1, out=buf57)
        del arg21_1
        buf58 = reinterpret_tensor(buf57, (1, 512, 3072), (1572864, 3072, 1), 0); del buf57  # reuse
        # Source Nodes: [add_10, add_11, hidden_states_20, mul_10, mul_8, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf58, arg20_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg20_1
        buf59 = reinterpret_tensor(buf56, (512, 768), (768, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (512, 3072), (3072, 1), 0), arg23_1, out=buf59)
        del arg23_1
        buf63 = reinterpret_tensor(buf39, (1, 512, 768), (393216, 768, 1), 0); del buf39  # reuse
        # Source Nodes: [hidden_states_24, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf52, buf59, arg22_1, arg62_1, arg63_1, buf63, 512, 768, grid=grid(512), stream=stream0)
        del arg62_1
        del arg63_1
        buf64 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg24_1, reinterpret_tensor(buf63, (512, 768), (768, 1), 0), arg25_1, alpha=1, beta=1, out=buf64)
        del arg24_1
        del arg25_1
        buf65 = reinterpret_tensor(buf48, (12, 512, 512), (262144, 512, 1), 0); del buf48  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf64, (12, 64, 512), (64, 1, 2304), 768), out=buf65)
        buf68 = reinterpret_tensor(buf45, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf45  # reuse
        # Source Nodes: [attn_weights_22, attn_weights_23, attn_weights_24, full_6, mask_value_3], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_1.run(arg80_1, buf65, buf68, 6144, 512, grid=grid(6144), stream=stream0)
        del arg80_1
        buf69 = reinterpret_tensor(buf63, (12, 512, 64), (32768, 64, 1), 0); del buf63  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf64, (12, 512, 64), (64, 2304, 1), 1536), out=buf69)
        buf70 = reinterpret_tensor(buf32, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf32  # reuse
        # Source Nodes: [tensor_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf69, buf70, 393216, grid=grid(393216), stream=stream0)
        buf71 = reinterpret_tensor(buf69, (512, 768), (768, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 768), (768, 1), 0), arg27_1, out=buf71)
        del arg27_1
        buf72 = reinterpret_tensor(buf71, (1, 512, 768), (393216, 768, 1), 0); del buf71  # reuse
        buf76 = reinterpret_tensor(buf70, (1, 512, 768), (393216, 768, 1), 0); del buf70  # reuse
        # Source Nodes: [hidden_states_26, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf72, arg26_1, buf52, buf59, arg22_1, arg64_1, arg65_1, buf76, 512, 768, grid=grid(512), stream=stream0)
        del arg22_1
        del arg26_1
        del arg64_1
        del arg65_1
        buf77 = reinterpret_tensor(buf58, (512, 3072), (3072, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 768), (768, 1), 0), arg29_1, out=buf77)
        del arg29_1
        buf78 = reinterpret_tensor(buf77, (1, 512, 3072), (1572864, 3072, 1), 0); del buf77  # reuse
        # Source Nodes: [add_14, add_15, hidden_states_28, mul_12, mul_13, mul_14, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf78, arg28_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg28_1
        buf79 = reinterpret_tensor(buf76, (512, 768), (768, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (512, 3072), (3072, 1), 0), arg31_1, out=buf79)
        del arg31_1
        buf83 = reinterpret_tensor(buf59, (1, 512, 768), (393216, 768, 1), 0); del buf59  # reuse
        # Source Nodes: [hidden_states_32, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf72, buf79, arg30_1, arg66_1, arg67_1, buf83, 512, 768, grid=grid(512), stream=stream0)
        del arg66_1
        del arg67_1
        buf84 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg32_1, reinterpret_tensor(buf83, (512, 768), (768, 1), 0), arg33_1, alpha=1, beta=1, out=buf84)
        del arg32_1
        del arg33_1
        buf85 = reinterpret_tensor(buf68, (12, 512, 512), (262144, 512, 1), 0); del buf68  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf84, (12, 64, 512), (64, 1, 2304), 768), out=buf85)
        buf88 = reinterpret_tensor(buf65, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf65  # reuse
        # Source Nodes: [attn_weights_29, attn_weights_30, attn_weights_31, full_8, mask_value_4], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_1.run(arg81_1, buf85, buf88, 6144, 512, grid=grid(6144), stream=stream0)
        del arg81_1
        buf89 = reinterpret_tensor(buf83, (12, 512, 64), (32768, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf84, (12, 512, 64), (64, 2304, 1), 1536), out=buf89)
        buf90 = reinterpret_tensor(buf52, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [tensor_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf89, buf90, 393216, grid=grid(393216), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (512, 768), (768, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (512, 768), (768, 1), 0), arg35_1, out=buf91)
        del arg35_1
        buf92 = reinterpret_tensor(buf91, (1, 512, 768), (393216, 768, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf90, (1, 512, 768), (393216, 768, 1), 0); del buf90  # reuse
        # Source Nodes: [hidden_states_34, residual_8, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf92, arg34_1, buf72, buf79, arg30_1, arg68_1, arg69_1, buf96, 512, 768, grid=grid(512), stream=stream0)
        del arg30_1
        del arg34_1
        del arg68_1
        del arg69_1
        buf97 = reinterpret_tensor(buf78, (512, 3072), (3072, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (512, 768), (768, 1), 0), arg37_1, out=buf97)
        del arg37_1
        buf98 = reinterpret_tensor(buf97, (1, 512, 3072), (1572864, 3072, 1), 0); del buf97  # reuse
        # Source Nodes: [add_18, add_19, hidden_states_36, mul_16, mul_17, mul_18, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf98, arg36_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg36_1
        buf99 = reinterpret_tensor(buf96, (512, 768), (768, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (512, 3072), (3072, 1), 0), arg39_1, out=buf99)
        del arg39_1
        buf103 = reinterpret_tensor(buf79, (1, 512, 768), (393216, 768, 1), 0); del buf79  # reuse
        # Source Nodes: [hidden_states_40, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf92, buf99, arg38_1, arg70_1, arg71_1, buf103, 512, 768, grid=grid(512), stream=stream0)
        del arg70_1
        del arg71_1
        buf104 = empty((512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg40_1, reinterpret_tensor(buf103, (512, 768), (768, 1), 0), arg41_1, alpha=1, beta=1, out=buf104)
        del arg40_1
        del arg41_1
        buf105 = reinterpret_tensor(buf88, (12, 512, 512), (262144, 512, 1), 0); del buf88  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (12, 512, 64), (64, 2304, 1), 0), reinterpret_tensor(buf104, (12, 64, 512), (64, 1, 2304), 768), out=buf105)
        buf108 = reinterpret_tensor(buf85, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf85  # reuse
        # Source Nodes: [attn_weights_36, attn_weights_37, attn_weights_38, full_10, mask_value_5], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_1.run(arg82_1, buf105, buf108, 6144, 512, grid=grid(6144), stream=stream0)
        del arg82_1
        del buf105
        buf109 = reinterpret_tensor(buf103, (12, 512, 64), (32768, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (12, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf104, (12, 512, 64), (64, 2304, 1), 1536), out=buf109)
        del buf108
        buf110 = reinterpret_tensor(buf72, (1, 512, 12, 64), (393216, 768, 64, 1), 0); del buf72  # reuse
        # Source Nodes: [tensor_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf109, buf110, 393216, grid=grid(393216), stream=stream0)
        buf111 = reinterpret_tensor(buf109, (512, 768), (768, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (512, 768), (768, 1), 0), arg43_1, out=buf111)
        del arg43_1
        buf112 = reinterpret_tensor(buf111, (1, 512, 768), (393216, 768, 1), 0); del buf111  # reuse
        buf116 = reinterpret_tensor(buf110, (1, 512, 768), (393216, 768, 1), 0); del buf110  # reuse
        # Source Nodes: [hidden_states_42, residual_10, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf112, arg42_1, buf92, buf99, arg38_1, arg72_1, arg73_1, buf116, 512, 768, grid=grid(512), stream=stream0)
        del arg38_1
        del arg42_1
        del arg72_1
        del arg73_1
        del buf92
        buf117 = reinterpret_tensor(buf98, (512, 3072), (3072, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (512, 768), (768, 1), 0), arg45_1, out=buf117)
        del arg45_1
        buf118 = reinterpret_tensor(buf117, (1, 512, 3072), (1572864, 3072, 1), 0); del buf117  # reuse
        # Source Nodes: [add_22, add_23, hidden_states_44, mul_20, mul_21, mul_22, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf118, arg44_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg44_1
        buf119 = reinterpret_tensor(buf116, (512, 768), (768, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (512, 3072), (3072, 1), 0), arg47_1, out=buf119)
        del arg47_1
        del buf118
        buf123 = reinterpret_tensor(buf99, (1, 512, 768), (393216, 768, 1), 0); del buf99  # reuse
        # Source Nodes: [hidden_states_47, l__mod___transformer_ln_f], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf112, buf119, arg46_1, arg74_1, arg75_1, buf123, 512, 768, grid=grid(512), stream=stream0)
        del arg46_1
        del arg74_1
        del arg75_1
        del buf112
        del buf119
        buf124 = empty((512, 50257), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 50257), (1, 768), 0), out=buf124)
        del arg76_1
        del buf123
        buf125 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        buf126 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_7.run(buf124, buf125, buf126, 511, 50257, grid=grid(511), stream=stream0)
        buf127 = empty((), device='cuda', dtype=torch.float32)
        buf129 = buf127; del buf127  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_8.run(buf129, arg84_1, buf124, buf125, buf126, 1, 511, grid=grid(1), stream=stream0)
        del arg84_1
        return (buf129, reinterpret_tensor(buf124, (1, 512, 50257), (25731584, 50257, 1), 0), reinterpret_tensor(buf4, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf4, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf24, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf24, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf44, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf44, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf64, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf64, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf84, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf84, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf104, (1, 12, 512, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf104, (1, 12, 512, 64), (0, 64, 2304, 1), 1536), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg78_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg79_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg80_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg81_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg82_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg83_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg84_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistillGPT2', benchmark_compiled_module)
