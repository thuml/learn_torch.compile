
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


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgry3k6krqzce2vtz6evlda25eh5nqnitncqlaf2oom22imoiox.py
# Source Nodes: [hidden_states, hidden_states_2, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# hidden_states => add
# hidden_states_2 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
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
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
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
        tmp5 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50257
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50257")
        tmp4 = tl.load(in_ptr1 + (r1 + (2048*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp15 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 + 50257
        tmp12 = tmp0 < 0
        tmp13 = tl.where(tmp12, tmp11, tmp0)
        tl.device_assert(((0 <= tmp13) & (tmp13 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp13 < 50257")
        tmp14 = tl.load(in_ptr1 + (r1 + (2048*tmp13)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = 2048.0
        tmp19 = tmp9 / tmp18
        tmp20 = 1e-05
        tmp21 = tmp19 + tmp20
        tmp22 = tl.math.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp27, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ms/cms4enii3lfgy7sma4mvev54x6owujoeoyji7f5hivws5wndspyc.py
# Source Nodes: [attn_weights_1, attn_weights_2, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.where]
# attn_weights_1 => where
# attn_weights_2 => amax, div, exp, sub_1, sum_1
# mask_value => full_default
triton_per_fused__softmax__to_copy_where_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_where_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, other=0.0)
    tmp2 = -3.4028234663852886e+38
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp9 / tmp13
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp14, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu26fqxtdg54gkufqu4qxd63cruoyd2ubphrktadmh5ogt2usdmv.py
# Source Nodes: [tensor_4], Original ATen: [aten.clone]
# tensor_4 => clone_2
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 16
    x2 = (xindex // 2048)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2) + (16384*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dmz4le7dd4kzkc4eu7yo7blvs4so3grvp2npri7kpndo7zknkh.py
# Source Nodes: [hidden_states, hidden_states_4, inputs_embeds, position_embeds, residual_1], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# hidden_states => add
# hidden_states_4 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# inputs_embeds => embedding
# position_embeds => embedding_1
# residual_1 => add_3
triton_red_fused_add_embedding_native_layer_norm_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3 + 50257
        tmp5 = tmp3 < 0
        tmp6 = tl.where(tmp5, tmp4, tmp3)
        tl.device_assert(((0 <= tmp6) & (tmp6 < 50257)) | ~xmask, "index out of bounds: 0 <= tmp6 < 50257")
        tmp7 = tl.load(in_ptr2 + (r1 + (2048*tmp6)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp2 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight,
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp10, rmask & xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight,
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp20 - tmp12
        tmp22 = 2048.0
        tmp23 = tmp18 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c64yvvoqwfwjt3qeda5gwm3hpdpuaeq4dum5z3rrnrmjqx3lvf63.py
# Source Nodes: [add_2, add_3, hidden_states_6, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_2 => add_6
# add_3 => add_7
# hidden_states_6 => mul_7
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8192
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


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7luwvco4wq35rlrjpg364ob2ucnpm23rzx2pgyvqxn2ntlxv2e.py
# Source Nodes: [hidden_states_11, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_11 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
# residual_2 => add_8
triton_red_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 2048.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndo46d6ezfmlhezvnzjh5w3a57oskukc6cstvydplokppmma3ty.py
# Source Nodes: [hidden_states_13, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_13 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
# residual_2 => add_8
# residual_3 => add_11
triton_red_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 + tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight,
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp18 - tmp10
        tmp20 = 2048.0
        tmp21 = tmp16 / tmp20
        tmp22 = 1e-05
        tmp23 = tmp21 + tmp22
        tmp24 = tl.math.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp29, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4brq3d7xtysqctwqybcnzobveb5xh3xezoajemdni3kh7ej4d6.py
# Source Nodes: [argmax, eq, long], Original ATen: [aten._to_copy, aten.argmax, aten.eq]
# argmax => argmax
# eq => eq
# long => convert_element_type_24
triton_red_fused__to_copy_argmax_eq_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_argmax_eq_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
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
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bxshdln2vdbhoyycp2s4jr42ncfbjftzxjhvha6byfwv5rymxm.py
# Source Nodes: [pooled_logits], Original ATen: [aten.index]
# pooled_logits => index
triton_poi_fused_index_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_8', 'mutated_arg_names': []},
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
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 - tmp2
    tmp4 = tmp3 + 128
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert((0 <= tmp6) & (tmp6 < 128), "index out of bounds: 0 <= tmp6 < 128")
    tmp7 = tl.load(in_ptr1 + (x0 + (2*tmp6)), xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1 = args
    args.clear()
    assert_size_stride(arg0_1, (50257, 2048), (2048, 1))
    assert_size_stride(arg1_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg2_1, (2048, ), (1, ))
    assert_size_stride(arg3_1, (2048, ), (1, ))
    assert_size_stride(arg4_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg5_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg6_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg7_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg8_1, (2048, ), (1, ))
    assert_size_stride(arg9_1, (2048, ), (1, ))
    assert_size_stride(arg10_1, (2048, ), (1, ))
    assert_size_stride(arg11_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg12_1, (8192, ), (1, ))
    assert_size_stride(arg13_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg14_1, (2048, ), (1, ))
    assert_size_stride(arg15_1, (2048, ), (1, ))
    assert_size_stride(arg16_1, (2048, ), (1, ))
    assert_size_stride(arg17_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg18_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg19_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg20_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg21_1, (2048, ), (1, ))
    assert_size_stride(arg22_1, (2048, ), (1, ))
    assert_size_stride(arg23_1, (2048, ), (1, ))
    assert_size_stride(arg24_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg25_1, (8192, ), (1, ))
    assert_size_stride(arg26_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg27_1, (2048, ), (1, ))
    assert_size_stride(arg28_1, (2048, ), (1, ))
    assert_size_stride(arg29_1, (2048, ), (1, ))
    assert_size_stride(arg30_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg31_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg32_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg33_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg34_1, (2048, ), (1, ))
    assert_size_stride(arg35_1, (2048, ), (1, ))
    assert_size_stride(arg36_1, (2048, ), (1, ))
    assert_size_stride(arg37_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg38_1, (8192, ), (1, ))
    assert_size_stride(arg39_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg40_1, (2048, ), (1, ))
    assert_size_stride(arg41_1, (2048, ), (1, ))
    assert_size_stride(arg42_1, (2048, ), (1, ))
    assert_size_stride(arg43_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg44_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg45_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg46_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg47_1, (2048, ), (1, ))
    assert_size_stride(arg48_1, (2048, ), (1, ))
    assert_size_stride(arg49_1, (2048, ), (1, ))
    assert_size_stride(arg50_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg51_1, (8192, ), (1, ))
    assert_size_stride(arg52_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg53_1, (2048, ), (1, ))
    assert_size_stride(arg54_1, (2048, ), (1, ))
    assert_size_stride(arg55_1, (2048, ), (1, ))
    assert_size_stride(arg56_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg57_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg58_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg59_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg60_1, (2048, ), (1, ))
    assert_size_stride(arg61_1, (2048, ), (1, ))
    assert_size_stride(arg62_1, (2048, ), (1, ))
    assert_size_stride(arg63_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg64_1, (8192, ), (1, ))
    assert_size_stride(arg65_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg66_1, (2048, ), (1, ))
    assert_size_stride(arg67_1, (2048, ), (1, ))
    assert_size_stride(arg68_1, (2048, ), (1, ))
    assert_size_stride(arg69_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg70_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg71_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg72_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg73_1, (2048, ), (1, ))
    assert_size_stride(arg74_1, (2048, ), (1, ))
    assert_size_stride(arg75_1, (2048, ), (1, ))
    assert_size_stride(arg76_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg77_1, (8192, ), (1, ))
    assert_size_stride(arg78_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg79_1, (2048, ), (1, ))
    assert_size_stride(arg80_1, (2048, ), (1, ))
    assert_size_stride(arg81_1, (2048, ), (1, ))
    assert_size_stride(arg82_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg83_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg84_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg85_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg86_1, (2048, ), (1, ))
    assert_size_stride(arg87_1, (2048, ), (1, ))
    assert_size_stride(arg88_1, (2048, ), (1, ))
    assert_size_stride(arg89_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg90_1, (8192, ), (1, ))
    assert_size_stride(arg91_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg92_1, (2048, ), (1, ))
    assert_size_stride(arg93_1, (2048, ), (1, ))
    assert_size_stride(arg94_1, (2048, ), (1, ))
    assert_size_stride(arg95_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg96_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg97_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg98_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg99_1, (2048, ), (1, ))
    assert_size_stride(arg100_1, (2048, ), (1, ))
    assert_size_stride(arg101_1, (2048, ), (1, ))
    assert_size_stride(arg102_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg103_1, (8192, ), (1, ))
    assert_size_stride(arg104_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg105_1, (2048, ), (1, ))
    assert_size_stride(arg106_1, (2048, ), (1, ))
    assert_size_stride(arg107_1, (2048, ), (1, ))
    assert_size_stride(arg108_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg109_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg110_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg111_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg112_1, (2048, ), (1, ))
    assert_size_stride(arg113_1, (2048, ), (1, ))
    assert_size_stride(arg114_1, (2048, ), (1, ))
    assert_size_stride(arg115_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg116_1, (8192, ), (1, ))
    assert_size_stride(arg117_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg118_1, (2048, ), (1, ))
    assert_size_stride(arg119_1, (2048, ), (1, ))
    assert_size_stride(arg120_1, (2048, ), (1, ))
    assert_size_stride(arg121_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg122_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg123_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg124_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg125_1, (2048, ), (1, ))
    assert_size_stride(arg126_1, (2048, ), (1, ))
    assert_size_stride(arg127_1, (2048, ), (1, ))
    assert_size_stride(arg128_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg129_1, (8192, ), (1, ))
    assert_size_stride(arg130_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg131_1, (2048, ), (1, ))
    assert_size_stride(arg132_1, (2048, ), (1, ))
    assert_size_stride(arg133_1, (2048, ), (1, ))
    assert_size_stride(arg134_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg135_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg136_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg137_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg138_1, (2048, ), (1, ))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (2048, ), (1, ))
    assert_size_stride(arg141_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg142_1, (8192, ), (1, ))
    assert_size_stride(arg143_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg144_1, (2048, ), (1, ))
    assert_size_stride(arg145_1, (2048, ), (1, ))
    assert_size_stride(arg146_1, (2048, ), (1, ))
    assert_size_stride(arg147_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg148_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg149_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg150_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg151_1, (2048, ), (1, ))
    assert_size_stride(arg152_1, (2048, ), (1, ))
    assert_size_stride(arg153_1, (2048, ), (1, ))
    assert_size_stride(arg154_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg155_1, (8192, ), (1, ))
    assert_size_stride(arg156_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg157_1, (2048, ), (1, ))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (2048, ), (1, ))
    assert_size_stride(arg160_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg161_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg162_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg163_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (2048, ), (1, ))
    assert_size_stride(arg166_1, (2048, ), (1, ))
    assert_size_stride(arg167_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg168_1, (8192, ), (1, ))
    assert_size_stride(arg169_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg170_1, (2048, ), (1, ))
    assert_size_stride(arg171_1, (2048, ), (1, ))
    assert_size_stride(arg172_1, (2048, ), (1, ))
    assert_size_stride(arg173_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg174_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg175_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg176_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg177_1, (2048, ), (1, ))
    assert_size_stride(arg178_1, (2048, ), (1, ))
    assert_size_stride(arg179_1, (2048, ), (1, ))
    assert_size_stride(arg180_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg181_1, (8192, ), (1, ))
    assert_size_stride(arg182_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg183_1, (2048, ), (1, ))
    assert_size_stride(arg184_1, (2048, ), (1, ))
    assert_size_stride(arg185_1, (2048, ), (1, ))
    assert_size_stride(arg186_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg187_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg188_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg189_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg190_1, (2048, ), (1, ))
    assert_size_stride(arg191_1, (2048, ), (1, ))
    assert_size_stride(arg192_1, (2048, ), (1, ))
    assert_size_stride(arg193_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg194_1, (8192, ), (1, ))
    assert_size_stride(arg195_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg196_1, (2048, ), (1, ))
    assert_size_stride(arg197_1, (2048, ), (1, ))
    assert_size_stride(arg198_1, (2048, ), (1, ))
    assert_size_stride(arg199_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg200_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg201_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg202_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg203_1, (2048, ), (1, ))
    assert_size_stride(arg204_1, (2048, ), (1, ))
    assert_size_stride(arg205_1, (2048, ), (1, ))
    assert_size_stride(arg206_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg207_1, (8192, ), (1, ))
    assert_size_stride(arg208_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg209_1, (2048, ), (1, ))
    assert_size_stride(arg210_1, (2048, ), (1, ))
    assert_size_stride(arg211_1, (2048, ), (1, ))
    assert_size_stride(arg212_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg213_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg214_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg215_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg216_1, (2048, ), (1, ))
    assert_size_stride(arg217_1, (2048, ), (1, ))
    assert_size_stride(arg218_1, (2048, ), (1, ))
    assert_size_stride(arg219_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg220_1, (8192, ), (1, ))
    assert_size_stride(arg221_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg222_1, (2048, ), (1, ))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (2048, ), (1, ))
    assert_size_stride(arg225_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg226_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg227_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg228_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg229_1, (2048, ), (1, ))
    assert_size_stride(arg230_1, (2048, ), (1, ))
    assert_size_stride(arg231_1, (2048, ), (1, ))
    assert_size_stride(arg232_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg233_1, (8192, ), (1, ))
    assert_size_stride(arg234_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg235_1, (2048, ), (1, ))
    assert_size_stride(arg236_1, (2048, ), (1, ))
    assert_size_stride(arg237_1, (2048, ), (1, ))
    assert_size_stride(arg238_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg239_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg240_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg241_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg242_1, (2048, ), (1, ))
    assert_size_stride(arg243_1, (2048, ), (1, ))
    assert_size_stride(arg244_1, (2048, ), (1, ))
    assert_size_stride(arg245_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg246_1, (8192, ), (1, ))
    assert_size_stride(arg247_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg248_1, (2048, ), (1, ))
    assert_size_stride(arg249_1, (2048, ), (1, ))
    assert_size_stride(arg250_1, (2048, ), (1, ))
    assert_size_stride(arg251_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg252_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg253_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg254_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg255_1, (2048, ), (1, ))
    assert_size_stride(arg256_1, (2048, ), (1, ))
    assert_size_stride(arg257_1, (2048, ), (1, ))
    assert_size_stride(arg258_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg259_1, (8192, ), (1, ))
    assert_size_stride(arg260_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg261_1, (2048, ), (1, ))
    assert_size_stride(arg262_1, (2048, ), (1, ))
    assert_size_stride(arg263_1, (2048, ), (1, ))
    assert_size_stride(arg264_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg265_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg266_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg267_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg268_1, (2048, ), (1, ))
    assert_size_stride(arg269_1, (2048, ), (1, ))
    assert_size_stride(arg270_1, (2048, ), (1, ))
    assert_size_stride(arg271_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg272_1, (8192, ), (1, ))
    assert_size_stride(arg273_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg274_1, (2048, ), (1, ))
    assert_size_stride(arg275_1, (2048, ), (1, ))
    assert_size_stride(arg276_1, (2048, ), (1, ))
    assert_size_stride(arg277_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg278_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg279_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg280_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg281_1, (2048, ), (1, ))
    assert_size_stride(arg282_1, (2048, ), (1, ))
    assert_size_stride(arg283_1, (2048, ), (1, ))
    assert_size_stride(arg284_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg285_1, (8192, ), (1, ))
    assert_size_stride(arg286_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg287_1, (2048, ), (1, ))
    assert_size_stride(arg288_1, (2048, ), (1, ))
    assert_size_stride(arg289_1, (2048, ), (1, ))
    assert_size_stride(arg290_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg291_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg292_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg293_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg294_1, (2048, ), (1, ))
    assert_size_stride(arg295_1, (2048, ), (1, ))
    assert_size_stride(arg296_1, (2048, ), (1, ))
    assert_size_stride(arg297_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg298_1, (8192, ), (1, ))
    assert_size_stride(arg299_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg300_1, (2048, ), (1, ))
    assert_size_stride(arg301_1, (2048, ), (1, ))
    assert_size_stride(arg302_1, (2048, ), (1, ))
    assert_size_stride(arg303_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg304_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg305_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg306_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg307_1, (2048, ), (1, ))
    assert_size_stride(arg308_1, (2048, ), (1, ))
    assert_size_stride(arg309_1, (2048, ), (1, ))
    assert_size_stride(arg310_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg311_1, (8192, ), (1, ))
    assert_size_stride(arg312_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg313_1, (2048, ), (1, ))
    assert_size_stride(arg314_1, (2048, ), (1, ))
    assert_size_stride(arg315_1, (2048, ), (1, ))
    assert_size_stride(arg316_1, (2, 2048), (2048, 1))
    assert_size_stride(arg317_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg318_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg319_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg320_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg321_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg322_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg323_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg324_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg325_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg326_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg327_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg328_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg329_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg330_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg331_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg332_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg333_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg334_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg335_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg336_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg337_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg338_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg339_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg340_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg341_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states, hidden_states_2, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_native_layer_norm_0.run(arg341_1, arg0_1, arg1_1, arg2_1, arg3_1, buf3, 128, 2048, grid=grid(128), stream=stream0)
        del arg2_1
        del arg3_1
        buf4 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [query], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg4_1, (2048, 2048), (1, 2048), 0), out=buf4)
        del arg4_1
        buf5 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg5_1, (2048, 2048), (1, 2048), 0), out=buf5)
        del arg5_1
        buf6 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf5, (16, 128, 128), (128, 1, 2048), 0), out=buf6)
        buf10 = reinterpret_tensor(buf4, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf4  # reuse
        # Source Nodes: [attn_weights_1, attn_weights_2, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg317_1, buf6, buf10, 2048, 128, grid=grid(2048), stream=stream0)
        del arg317_1
        buf9 = reinterpret_tensor(buf6, (128, 2048), (2048, 1), 0); del buf6  # reuse
        # Source Nodes: [value], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg6_1, (2048, 2048), (1, 2048), 0), out=buf9)
        del arg6_1
        buf11 = reinterpret_tensor(buf3, (16, 128, 128), (16384, 128, 1), 0); del buf3  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf9, (16, 128, 128), (128, 2048, 1), 0), out=buf11)
        buf12 = reinterpret_tensor(buf10, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf10  # reuse
        # Source Nodes: [tensor_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf11, buf12, 262144, grid=grid(262144), stream=stream0)
        buf13 = reinterpret_tensor(buf11, (128, 2048), (2048, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf12, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg7_1, (2048, 2048), (1, 2048), 0), out=buf13)
        del arg7_1
        buf14 = reinterpret_tensor(buf13, (1, 128, 2048), (262144, 2048, 1), 0); del buf13  # reuse
        buf18 = reinterpret_tensor(buf12, (1, 128, 2048), (262144, 2048, 1), 0); del buf12  # reuse
        # Source Nodes: [hidden_states, hidden_states_4, inputs_embeds, position_embeds, residual_1], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        triton_red_fused_add_embedding_native_layer_norm_3.run(buf14, arg8_1, arg341_1, arg0_1, arg1_1, arg9_1, arg10_1, buf18, 128, 2048, grid=grid(128), stream=stream0)
        del arg0_1
        del arg10_1
        del arg1_1
        del arg8_1
        del arg9_1
        buf19 = empty((128, 8192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg11_1, (2048, 8192), (1, 2048), 0), out=buf19)
        del arg11_1
        buf20 = reinterpret_tensor(buf19, (1, 128, 8192), (1048576, 8192, 1), 0); del buf19  # reuse
        # Source Nodes: [add_2, add_3, hidden_states_6, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf20, arg12_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg12_1
        buf21 = reinterpret_tensor(buf18, (128, 2048), (2048, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg13_1, (8192, 2048), (1, 8192), 0), out=buf21)
        del arg13_1
        buf25 = empty((1, 128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_11, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf14, buf21, arg14_1, arg15_1, arg16_1, buf25, 128, 2048, grid=grid(128), stream=stream0)
        del arg15_1
        del arg16_1
        buf26 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [query_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg17_1, (2048, 2048), (1, 2048), 0), out=buf26)
        del arg17_1
        buf27 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg18_1, (2048, 2048), (1, 2048), 0), out=buf27)
        del arg18_1
        buf28 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf27, (16, 128, 128), (128, 1, 2048), 0), out=buf28)
        buf32 = reinterpret_tensor(buf26, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf26  # reuse
        # Source Nodes: [attn_weights_7, attn_weights_8, mask_value_1], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg318_1, buf28, buf32, 2048, 128, grid=grid(2048), stream=stream0)
        del arg318_1
        buf31 = reinterpret_tensor(buf28, (128, 2048), (2048, 1), 0); del buf28  # reuse
        # Source Nodes: [value_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg19_1, (2048, 2048), (1, 2048), 0), out=buf31)
        del arg19_1
        buf33 = reinterpret_tensor(buf25, (16, 128, 128), (16384, 128, 1), 0); del buf25  # reuse
        # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf31, (16, 128, 128), (128, 2048, 1), 0), out=buf33)
        buf34 = reinterpret_tensor(buf32, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf32  # reuse
        # Source Nodes: [tensor_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf33, buf34, 262144, grid=grid(262144), stream=stream0)
        buf35 = reinterpret_tensor(buf33, (128, 2048), (2048, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg20_1, (2048, 2048), (1, 2048), 0), out=buf35)
        del arg20_1
        buf36 = reinterpret_tensor(buf35, (1, 128, 2048), (262144, 2048, 1), 0); del buf35  # reuse
        buf40 = reinterpret_tensor(buf34, (1, 128, 2048), (262144, 2048, 1), 0); del buf34  # reuse
        # Source Nodes: [hidden_states_13, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf36, arg21_1, buf14, buf21, arg14_1, arg22_1, arg23_1, buf40, 128, 2048, grid=grid(128), stream=stream0)
        del arg14_1
        del arg21_1
        del arg22_1
        del arg23_1
        buf41 = reinterpret_tensor(buf20, (128, 8192), (8192, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg24_1, (2048, 8192), (1, 2048), 0), out=buf41)
        del arg24_1
        buf42 = reinterpret_tensor(buf41, (1, 128, 8192), (1048576, 8192, 1), 0); del buf41  # reuse
        # Source Nodes: [add_6, add_7, hidden_states_15, mul_4, mul_5, mul_6, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf42, arg25_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg25_1
        buf43 = reinterpret_tensor(buf40, (128, 2048), (2048, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg26_1, (8192, 2048), (1, 8192), 0), out=buf43)
        del arg26_1
        buf47 = reinterpret_tensor(buf21, (1, 128, 2048), (262144, 2048, 1), 0); del buf21  # reuse
        # Source Nodes: [hidden_states_20, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf36, buf43, arg27_1, arg28_1, arg29_1, buf47, 128, 2048, grid=grid(128), stream=stream0)
        del arg28_1
        del arg29_1
        buf48 = reinterpret_tensor(buf14, (128, 2048), (2048, 1), 0); del buf14  # reuse
        # Source Nodes: [query_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg30_1, (2048, 2048), (1, 2048), 0), out=buf48)
        del arg30_1
        buf49 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg31_1, (2048, 2048), (1, 2048), 0), out=buf49)
        del arg31_1
        buf50 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf49, (16, 128, 128), (128, 1, 2048), 0), out=buf50)
        buf54 = reinterpret_tensor(buf48, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf48  # reuse
        # Source Nodes: [attn_weights_13, attn_weights_14, mask_value_2], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg319_1, buf50, buf54, 2048, 128, grid=grid(2048), stream=stream0)
        del arg319_1
        buf53 = reinterpret_tensor(buf50, (128, 2048), (2048, 1), 0); del buf50  # reuse
        # Source Nodes: [value_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg32_1, (2048, 2048), (1, 2048), 0), out=buf53)
        del arg32_1
        buf55 = reinterpret_tensor(buf47, (16, 128, 128), (16384, 128, 1), 0); del buf47  # reuse
        # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf53, (16, 128, 128), (128, 2048, 1), 0), out=buf55)
        buf56 = reinterpret_tensor(buf54, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf54  # reuse
        # Source Nodes: [tensor_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf55, buf56, 262144, grid=grid(262144), stream=stream0)
        buf57 = reinterpret_tensor(buf55, (128, 2048), (2048, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg33_1, (2048, 2048), (1, 2048), 0), out=buf57)
        del arg33_1
        buf58 = reinterpret_tensor(buf57, (1, 128, 2048), (262144, 2048, 1), 0); del buf57  # reuse
        buf62 = reinterpret_tensor(buf56, (1, 128, 2048), (262144, 2048, 1), 0); del buf56  # reuse
        # Source Nodes: [hidden_states_22, residual_4, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf58, arg34_1, buf36, buf43, arg27_1, arg35_1, arg36_1, buf62, 128, 2048, grid=grid(128), stream=stream0)
        del arg27_1
        del arg34_1
        del arg35_1
        del arg36_1
        buf63 = reinterpret_tensor(buf42, (128, 8192), (8192, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf62, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg37_1, (2048, 8192), (1, 2048), 0), out=buf63)
        del arg37_1
        buf64 = reinterpret_tensor(buf63, (1, 128, 8192), (1048576, 8192, 1), 0); del buf63  # reuse
        # Source Nodes: [add_10, add_11, hidden_states_24, mul_10, mul_8, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf64, arg38_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg38_1
        buf65 = reinterpret_tensor(buf62, (128, 2048), (2048, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg39_1, (8192, 2048), (1, 8192), 0), out=buf65)
        del arg39_1
        buf69 = reinterpret_tensor(buf43, (1, 128, 2048), (262144, 2048, 1), 0); del buf43  # reuse
        # Source Nodes: [hidden_states_29, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf58, buf65, arg40_1, arg41_1, arg42_1, buf69, 128, 2048, grid=grid(128), stream=stream0)
        del arg41_1
        del arg42_1
        buf70 = reinterpret_tensor(buf36, (128, 2048), (2048, 1), 0); del buf36  # reuse
        # Source Nodes: [query_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg43_1, (2048, 2048), (1, 2048), 0), out=buf70)
        del arg43_1
        buf71 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg44_1, (2048, 2048), (1, 2048), 0), out=buf71)
        del arg44_1
        buf72 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf71, (16, 128, 128), (128, 1, 2048), 0), out=buf72)
        buf76 = reinterpret_tensor(buf70, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf70  # reuse
        # Source Nodes: [attn_weights_19, attn_weights_20, mask_value_3], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg320_1, buf72, buf76, 2048, 128, grid=grid(2048), stream=stream0)
        del arg320_1
        buf75 = reinterpret_tensor(buf72, (128, 2048), (2048, 1), 0); del buf72  # reuse
        # Source Nodes: [value_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg45_1, (2048, 2048), (1, 2048), 0), out=buf75)
        del arg45_1
        buf77 = reinterpret_tensor(buf69, (16, 128, 128), (16384, 128, 1), 0); del buf69  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf75, (16, 128, 128), (128, 2048, 1), 0), out=buf77)
        buf78 = reinterpret_tensor(buf76, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf76  # reuse
        # Source Nodes: [tensor_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf77, buf78, 262144, grid=grid(262144), stream=stream0)
        buf79 = reinterpret_tensor(buf77, (128, 2048), (2048, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg46_1, (2048, 2048), (1, 2048), 0), out=buf79)
        del arg46_1
        buf80 = reinterpret_tensor(buf79, (1, 128, 2048), (262144, 2048, 1), 0); del buf79  # reuse
        buf84 = reinterpret_tensor(buf78, (1, 128, 2048), (262144, 2048, 1), 0); del buf78  # reuse
        # Source Nodes: [hidden_states_31, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf80, arg47_1, buf58, buf65, arg40_1, arg48_1, arg49_1, buf84, 128, 2048, grid=grid(128), stream=stream0)
        del arg40_1
        del arg47_1
        del arg48_1
        del arg49_1
        buf85 = reinterpret_tensor(buf64, (128, 8192), (8192, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg50_1, (2048, 8192), (1, 2048), 0), out=buf85)
        del arg50_1
        buf86 = reinterpret_tensor(buf85, (1, 128, 8192), (1048576, 8192, 1), 0); del buf85  # reuse
        # Source Nodes: [add_14, add_15, hidden_states_33, mul_12, mul_13, mul_14, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf86, arg51_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg51_1
        buf87 = reinterpret_tensor(buf84, (128, 2048), (2048, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf86, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg52_1, (8192, 2048), (1, 8192), 0), out=buf87)
        del arg52_1
        buf91 = reinterpret_tensor(buf65, (1, 128, 2048), (262144, 2048, 1), 0); del buf65  # reuse
        # Source Nodes: [hidden_states_38, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf80, buf87, arg53_1, arg54_1, arg55_1, buf91, 128, 2048, grid=grid(128), stream=stream0)
        del arg54_1
        del arg55_1
        buf92 = reinterpret_tensor(buf58, (128, 2048), (2048, 1), 0); del buf58  # reuse
        # Source Nodes: [query_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg56_1, (2048, 2048), (1, 2048), 0), out=buf92)
        del arg56_1
        buf93 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg57_1, (2048, 2048), (1, 2048), 0), out=buf93)
        del arg57_1
        buf94 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf92, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf93, (16, 128, 128), (128, 1, 2048), 0), out=buf94)
        buf98 = reinterpret_tensor(buf92, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf92  # reuse
        # Source Nodes: [attn_weights_25, attn_weights_26, mask_value_4], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg321_1, buf94, buf98, 2048, 128, grid=grid(2048), stream=stream0)
        del arg321_1
        buf97 = reinterpret_tensor(buf94, (128, 2048), (2048, 1), 0); del buf94  # reuse
        # Source Nodes: [value_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg58_1, (2048, 2048), (1, 2048), 0), out=buf97)
        del arg58_1
        buf99 = reinterpret_tensor(buf91, (16, 128, 128), (16384, 128, 1), 0); del buf91  # reuse
        # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf97, (16, 128, 128), (128, 2048, 1), 0), out=buf99)
        buf100 = reinterpret_tensor(buf98, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf98  # reuse
        # Source Nodes: [tensor_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf99, buf100, 262144, grid=grid(262144), stream=stream0)
        buf101 = reinterpret_tensor(buf99, (128, 2048), (2048, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg59_1, (2048, 2048), (1, 2048), 0), out=buf101)
        del arg59_1
        buf102 = reinterpret_tensor(buf101, (1, 128, 2048), (262144, 2048, 1), 0); del buf101  # reuse
        buf106 = reinterpret_tensor(buf100, (1, 128, 2048), (262144, 2048, 1), 0); del buf100  # reuse
        # Source Nodes: [hidden_states_40, residual_8, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf102, arg60_1, buf80, buf87, arg53_1, arg61_1, arg62_1, buf106, 128, 2048, grid=grid(128), stream=stream0)
        del arg53_1
        del arg60_1
        del arg61_1
        del arg62_1
        buf107 = reinterpret_tensor(buf86, (128, 8192), (8192, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 8192), (1, 2048), 0), out=buf107)
        del arg63_1
        buf108 = reinterpret_tensor(buf107, (1, 128, 8192), (1048576, 8192, 1), 0); del buf107  # reuse
        # Source Nodes: [add_18, add_19, hidden_states_42, mul_16, mul_17, mul_18, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf108, arg64_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg64_1
        buf109 = reinterpret_tensor(buf106, (128, 2048), (2048, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg65_1, (8192, 2048), (1, 8192), 0), out=buf109)
        del arg65_1
        buf113 = reinterpret_tensor(buf87, (1, 128, 2048), (262144, 2048, 1), 0); del buf87  # reuse
        # Source Nodes: [hidden_states_47, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf102, buf109, arg66_1, arg67_1, arg68_1, buf113, 128, 2048, grid=grid(128), stream=stream0)
        del arg67_1
        del arg68_1
        buf114 = reinterpret_tensor(buf80, (128, 2048), (2048, 1), 0); del buf80  # reuse
        # Source Nodes: [query_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg69_1, (2048, 2048), (1, 2048), 0), out=buf114)
        del arg69_1
        buf115 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg70_1, (2048, 2048), (1, 2048), 0), out=buf115)
        del arg70_1
        buf116 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf115, (16, 128, 128), (128, 1, 2048), 0), out=buf116)
        buf120 = reinterpret_tensor(buf114, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf114  # reuse
        # Source Nodes: [attn_weights_31, attn_weights_32, mask_value_5], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg322_1, buf116, buf120, 2048, 128, grid=grid(2048), stream=stream0)
        del arg322_1
        buf119 = reinterpret_tensor(buf116, (128, 2048), (2048, 1), 0); del buf116  # reuse
        # Source Nodes: [value_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg71_1, (2048, 2048), (1, 2048), 0), out=buf119)
        del arg71_1
        buf121 = reinterpret_tensor(buf113, (16, 128, 128), (16384, 128, 1), 0); del buf113  # reuse
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf119, (16, 128, 128), (128, 2048, 1), 0), out=buf121)
        buf122 = reinterpret_tensor(buf120, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf120  # reuse
        # Source Nodes: [tensor_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf121, buf122, 262144, grid=grid(262144), stream=stream0)
        buf123 = reinterpret_tensor(buf121, (128, 2048), (2048, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg72_1, (2048, 2048), (1, 2048), 0), out=buf123)
        del arg72_1
        buf124 = reinterpret_tensor(buf123, (1, 128, 2048), (262144, 2048, 1), 0); del buf123  # reuse
        buf128 = reinterpret_tensor(buf122, (1, 128, 2048), (262144, 2048, 1), 0); del buf122  # reuse
        # Source Nodes: [hidden_states_49, residual_10, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf124, arg73_1, buf102, buf109, arg66_1, arg74_1, arg75_1, buf128, 128, 2048, grid=grid(128), stream=stream0)
        del arg66_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf129 = reinterpret_tensor(buf108, (128, 8192), (8192, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg76_1, (2048, 8192), (1, 2048), 0), out=buf129)
        del arg76_1
        buf130 = reinterpret_tensor(buf129, (1, 128, 8192), (1048576, 8192, 1), 0); del buf129  # reuse
        # Source Nodes: [add_22, add_23, hidden_states_51, mul_20, mul_21, mul_22, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf130, arg77_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg77_1
        buf131 = reinterpret_tensor(buf128, (128, 2048), (2048, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg78_1, (8192, 2048), (1, 8192), 0), out=buf131)
        del arg78_1
        buf135 = reinterpret_tensor(buf109, (1, 128, 2048), (262144, 2048, 1), 0); del buf109  # reuse
        # Source Nodes: [hidden_states_56, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf124, buf131, arg79_1, arg80_1, arg81_1, buf135, 128, 2048, grid=grid(128), stream=stream0)
        del arg80_1
        del arg81_1
        buf136 = reinterpret_tensor(buf102, (128, 2048), (2048, 1), 0); del buf102  # reuse
        # Source Nodes: [query_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg82_1, (2048, 2048), (1, 2048), 0), out=buf136)
        del arg82_1
        buf137 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg83_1, (2048, 2048), (1, 2048), 0), out=buf137)
        del arg83_1
        buf138 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf137, (16, 128, 128), (128, 1, 2048), 0), out=buf138)
        buf142 = reinterpret_tensor(buf136, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf136  # reuse
        # Source Nodes: [attn_weights_37, attn_weights_38, mask_value_6], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg323_1, buf138, buf142, 2048, 128, grid=grid(2048), stream=stream0)
        del arg323_1
        buf141 = reinterpret_tensor(buf138, (128, 2048), (2048, 1), 0); del buf138  # reuse
        # Source Nodes: [value_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg84_1, (2048, 2048), (1, 2048), 0), out=buf141)
        del arg84_1
        buf143 = reinterpret_tensor(buf135, (16, 128, 128), (16384, 128, 1), 0); del buf135  # reuse
        # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf141, (16, 128, 128), (128, 2048, 1), 0), out=buf143)
        buf144 = reinterpret_tensor(buf142, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf142  # reuse
        # Source Nodes: [tensor_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf143, buf144, 262144, grid=grid(262144), stream=stream0)
        buf145 = reinterpret_tensor(buf143, (128, 2048), (2048, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg85_1, (2048, 2048), (1, 2048), 0), out=buf145)
        del arg85_1
        buf146 = reinterpret_tensor(buf145, (1, 128, 2048), (262144, 2048, 1), 0); del buf145  # reuse
        buf150 = reinterpret_tensor(buf144, (1, 128, 2048), (262144, 2048, 1), 0); del buf144  # reuse
        # Source Nodes: [hidden_states_58, residual_12, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf146, arg86_1, buf124, buf131, arg79_1, arg87_1, arg88_1, buf150, 128, 2048, grid=grid(128), stream=stream0)
        del arg79_1
        del arg86_1
        del arg87_1
        del arg88_1
        buf151 = reinterpret_tensor(buf130, (128, 8192), (8192, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf150, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg89_1, (2048, 8192), (1, 2048), 0), out=buf151)
        del arg89_1
        buf152 = reinterpret_tensor(buf151, (1, 128, 8192), (1048576, 8192, 1), 0); del buf151  # reuse
        # Source Nodes: [add_26, add_27, hidden_states_60, mul_24, mul_25, mul_26, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf152, arg90_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg90_1
        buf153 = reinterpret_tensor(buf150, (128, 2048), (2048, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg91_1, (8192, 2048), (1, 8192), 0), out=buf153)
        del arg91_1
        buf157 = reinterpret_tensor(buf131, (1, 128, 2048), (262144, 2048, 1), 0); del buf131  # reuse
        # Source Nodes: [hidden_states_65, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf146, buf153, arg92_1, arg93_1, arg94_1, buf157, 128, 2048, grid=grid(128), stream=stream0)
        del arg93_1
        del arg94_1
        buf158 = reinterpret_tensor(buf124, (128, 2048), (2048, 1), 0); del buf124  # reuse
        # Source Nodes: [query_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg95_1, (2048, 2048), (1, 2048), 0), out=buf158)
        del arg95_1
        buf159 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg96_1, (2048, 2048), (1, 2048), 0), out=buf159)
        del arg96_1
        buf160 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf159, (16, 128, 128), (128, 1, 2048), 0), out=buf160)
        buf164 = reinterpret_tensor(buf158, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf158  # reuse
        # Source Nodes: [attn_weights_43, attn_weights_44, mask_value_7], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg324_1, buf160, buf164, 2048, 128, grid=grid(2048), stream=stream0)
        del arg324_1
        buf163 = reinterpret_tensor(buf160, (128, 2048), (2048, 1), 0); del buf160  # reuse
        # Source Nodes: [value_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg97_1, (2048, 2048), (1, 2048), 0), out=buf163)
        del arg97_1
        buf165 = reinterpret_tensor(buf157, (16, 128, 128), (16384, 128, 1), 0); del buf157  # reuse
        # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf163, (16, 128, 128), (128, 2048, 1), 0), out=buf165)
        buf166 = reinterpret_tensor(buf164, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf164  # reuse
        # Source Nodes: [tensor_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf165, buf166, 262144, grid=grid(262144), stream=stream0)
        buf167 = reinterpret_tensor(buf165, (128, 2048), (2048, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg98_1, (2048, 2048), (1, 2048), 0), out=buf167)
        del arg98_1
        buf168 = reinterpret_tensor(buf167, (1, 128, 2048), (262144, 2048, 1), 0); del buf167  # reuse
        buf172 = reinterpret_tensor(buf166, (1, 128, 2048), (262144, 2048, 1), 0); del buf166  # reuse
        # Source Nodes: [hidden_states_67, residual_14, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf168, arg99_1, buf146, buf153, arg92_1, arg100_1, arg101_1, buf172, 128, 2048, grid=grid(128), stream=stream0)
        del arg100_1
        del arg101_1
        del arg92_1
        del arg99_1
        buf173 = reinterpret_tensor(buf152, (128, 8192), (8192, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg102_1, (2048, 8192), (1, 2048), 0), out=buf173)
        del arg102_1
        buf174 = reinterpret_tensor(buf173, (1, 128, 8192), (1048576, 8192, 1), 0); del buf173  # reuse
        # Source Nodes: [add_30, add_31, hidden_states_69, mul_28, mul_29, mul_30, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf174, arg103_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg103_1
        buf175 = reinterpret_tensor(buf172, (128, 2048), (2048, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg104_1, (8192, 2048), (1, 8192), 0), out=buf175)
        del arg104_1
        buf179 = reinterpret_tensor(buf153, (1, 128, 2048), (262144, 2048, 1), 0); del buf153  # reuse
        # Source Nodes: [hidden_states_74, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf168, buf175, arg105_1, arg106_1, arg107_1, buf179, 128, 2048, grid=grid(128), stream=stream0)
        del arg106_1
        del arg107_1
        buf180 = reinterpret_tensor(buf146, (128, 2048), (2048, 1), 0); del buf146  # reuse
        # Source Nodes: [query_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg108_1, (2048, 2048), (1, 2048), 0), out=buf180)
        del arg108_1
        buf181 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg109_1, (2048, 2048), (1, 2048), 0), out=buf181)
        del arg109_1
        buf182 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf181, (16, 128, 128), (128, 1, 2048), 0), out=buf182)
        buf186 = reinterpret_tensor(buf180, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf180  # reuse
        # Source Nodes: [attn_weights_49, attn_weights_50, mask_value_8], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg325_1, buf182, buf186, 2048, 128, grid=grid(2048), stream=stream0)
        del arg325_1
        buf185 = reinterpret_tensor(buf182, (128, 2048), (2048, 1), 0); del buf182  # reuse
        # Source Nodes: [value_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 2048), (1, 2048), 0), out=buf185)
        del arg110_1
        buf187 = reinterpret_tensor(buf179, (16, 128, 128), (16384, 128, 1), 0); del buf179  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf185, (16, 128, 128), (128, 2048, 1), 0), out=buf187)
        buf188 = reinterpret_tensor(buf186, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf186  # reuse
        # Source Nodes: [tensor_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf187, buf188, 262144, grid=grid(262144), stream=stream0)
        buf189 = reinterpret_tensor(buf187, (128, 2048), (2048, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg111_1, (2048, 2048), (1, 2048), 0), out=buf189)
        del arg111_1
        buf190 = reinterpret_tensor(buf189, (1, 128, 2048), (262144, 2048, 1), 0); del buf189  # reuse
        buf194 = reinterpret_tensor(buf188, (1, 128, 2048), (262144, 2048, 1), 0); del buf188  # reuse
        # Source Nodes: [hidden_states_76, residual_16, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf190, arg112_1, buf168, buf175, arg105_1, arg113_1, arg114_1, buf194, 128, 2048, grid=grid(128), stream=stream0)
        del arg105_1
        del arg112_1
        del arg113_1
        del arg114_1
        buf195 = reinterpret_tensor(buf174, (128, 8192), (8192, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg115_1, (2048, 8192), (1, 2048), 0), out=buf195)
        del arg115_1
        buf196 = reinterpret_tensor(buf195, (1, 128, 8192), (1048576, 8192, 1), 0); del buf195  # reuse
        # Source Nodes: [add_34, add_35, hidden_states_78, mul_32, mul_33, mul_34, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf196, arg116_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg116_1
        buf197 = reinterpret_tensor(buf194, (128, 2048), (2048, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg117_1, (8192, 2048), (1, 8192), 0), out=buf197)
        del arg117_1
        buf201 = reinterpret_tensor(buf175, (1, 128, 2048), (262144, 2048, 1), 0); del buf175  # reuse
        # Source Nodes: [hidden_states_83, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf190, buf197, arg118_1, arg119_1, arg120_1, buf201, 128, 2048, grid=grid(128), stream=stream0)
        del arg119_1
        del arg120_1
        buf202 = reinterpret_tensor(buf168, (128, 2048), (2048, 1), 0); del buf168  # reuse
        # Source Nodes: [query_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg121_1, (2048, 2048), (1, 2048), 0), out=buf202)
        del arg121_1
        buf203 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg122_1, (2048, 2048), (1, 2048), 0), out=buf203)
        del arg122_1
        buf204 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf202, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf203, (16, 128, 128), (128, 1, 2048), 0), out=buf204)
        buf208 = reinterpret_tensor(buf202, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf202  # reuse
        # Source Nodes: [attn_weights_55, attn_weights_56, mask_value_9], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg326_1, buf204, buf208, 2048, 128, grid=grid(2048), stream=stream0)
        del arg326_1
        buf207 = reinterpret_tensor(buf204, (128, 2048), (2048, 1), 0); del buf204  # reuse
        # Source Nodes: [value_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg123_1, (2048, 2048), (1, 2048), 0), out=buf207)
        del arg123_1
        buf209 = reinterpret_tensor(buf201, (16, 128, 128), (16384, 128, 1), 0); del buf201  # reuse
        # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf207, (16, 128, 128), (128, 2048, 1), 0), out=buf209)
        buf210 = reinterpret_tensor(buf208, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf208  # reuse
        # Source Nodes: [tensor_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf209, buf210, 262144, grid=grid(262144), stream=stream0)
        buf211 = reinterpret_tensor(buf209, (128, 2048), (2048, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg124_1, (2048, 2048), (1, 2048), 0), out=buf211)
        del arg124_1
        buf212 = reinterpret_tensor(buf211, (1, 128, 2048), (262144, 2048, 1), 0); del buf211  # reuse
        buf216 = reinterpret_tensor(buf210, (1, 128, 2048), (262144, 2048, 1), 0); del buf210  # reuse
        # Source Nodes: [hidden_states_85, residual_18, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf212, arg125_1, buf190, buf197, arg118_1, arg126_1, arg127_1, buf216, 128, 2048, grid=grid(128), stream=stream0)
        del arg118_1
        del arg125_1
        del arg126_1
        del arg127_1
        buf217 = reinterpret_tensor(buf196, (128, 8192), (8192, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg128_1, (2048, 8192), (1, 2048), 0), out=buf217)
        del arg128_1
        buf218 = reinterpret_tensor(buf217, (1, 128, 8192), (1048576, 8192, 1), 0); del buf217  # reuse
        # Source Nodes: [add_38, add_39, hidden_states_87, mul_36, mul_37, mul_38, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf218, arg129_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg129_1
        buf219 = reinterpret_tensor(buf216, (128, 2048), (2048, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg130_1, (8192, 2048), (1, 8192), 0), out=buf219)
        del arg130_1
        buf223 = reinterpret_tensor(buf197, (1, 128, 2048), (262144, 2048, 1), 0); del buf197  # reuse
        # Source Nodes: [hidden_states_92, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf212, buf219, arg131_1, arg132_1, arg133_1, buf223, 128, 2048, grid=grid(128), stream=stream0)
        del arg132_1
        del arg133_1
        buf224 = reinterpret_tensor(buf190, (128, 2048), (2048, 1), 0); del buf190  # reuse
        # Source Nodes: [query_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg134_1, (2048, 2048), (1, 2048), 0), out=buf224)
        del arg134_1
        buf225 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg135_1, (2048, 2048), (1, 2048), 0), out=buf225)
        del arg135_1
        buf226 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf225, (16, 128, 128), (128, 1, 2048), 0), out=buf226)
        buf230 = reinterpret_tensor(buf224, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf224  # reuse
        # Source Nodes: [attn_weights_61, attn_weights_62, mask_value_10], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg327_1, buf226, buf230, 2048, 128, grid=grid(2048), stream=stream0)
        del arg327_1
        buf229 = reinterpret_tensor(buf226, (128, 2048), (2048, 1), 0); del buf226  # reuse
        # Source Nodes: [value_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg136_1, (2048, 2048), (1, 2048), 0), out=buf229)
        del arg136_1
        buf231 = reinterpret_tensor(buf223, (16, 128, 128), (16384, 128, 1), 0); del buf223  # reuse
        # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf230, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf229, (16, 128, 128), (128, 2048, 1), 0), out=buf231)
        buf232 = reinterpret_tensor(buf230, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf230  # reuse
        # Source Nodes: [tensor_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf231, buf232, 262144, grid=grid(262144), stream=stream0)
        buf233 = reinterpret_tensor(buf231, (128, 2048), (2048, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg137_1, (2048, 2048), (1, 2048), 0), out=buf233)
        del arg137_1
        buf234 = reinterpret_tensor(buf233, (1, 128, 2048), (262144, 2048, 1), 0); del buf233  # reuse
        buf238 = reinterpret_tensor(buf232, (1, 128, 2048), (262144, 2048, 1), 0); del buf232  # reuse
        # Source Nodes: [hidden_states_94, residual_20, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf234, arg138_1, buf212, buf219, arg131_1, arg139_1, arg140_1, buf238, 128, 2048, grid=grid(128), stream=stream0)
        del arg131_1
        del arg138_1
        del arg139_1
        del arg140_1
        buf239 = reinterpret_tensor(buf218, (128, 8192), (8192, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg141_1, (2048, 8192), (1, 2048), 0), out=buf239)
        del arg141_1
        buf240 = reinterpret_tensor(buf239, (1, 128, 8192), (1048576, 8192, 1), 0); del buf239  # reuse
        # Source Nodes: [add_42, add_43, hidden_states_96, mul_40, mul_41, mul_42, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf240, arg142_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg142_1
        buf241 = reinterpret_tensor(buf238, (128, 2048), (2048, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg143_1, (8192, 2048), (1, 8192), 0), out=buf241)
        del arg143_1
        buf245 = reinterpret_tensor(buf219, (1, 128, 2048), (262144, 2048, 1), 0); del buf219  # reuse
        # Source Nodes: [hidden_states_101, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf234, buf241, arg144_1, arg145_1, arg146_1, buf245, 128, 2048, grid=grid(128), stream=stream0)
        del arg145_1
        del arg146_1
        buf246 = reinterpret_tensor(buf212, (128, 2048), (2048, 1), 0); del buf212  # reuse
        # Source Nodes: [query_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg147_1, (2048, 2048), (1, 2048), 0), out=buf246)
        del arg147_1
        buf247 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg148_1, (2048, 2048), (1, 2048), 0), out=buf247)
        del arg148_1
        buf248 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf247, (16, 128, 128), (128, 1, 2048), 0), out=buf248)
        buf252 = reinterpret_tensor(buf246, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf246  # reuse
        # Source Nodes: [attn_weights_67, attn_weights_68, mask_value_11], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg328_1, buf248, buf252, 2048, 128, grid=grid(2048), stream=stream0)
        del arg328_1
        buf251 = reinterpret_tensor(buf248, (128, 2048), (2048, 1), 0); del buf248  # reuse
        # Source Nodes: [value_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg149_1, (2048, 2048), (1, 2048), 0), out=buf251)
        del arg149_1
        buf253 = reinterpret_tensor(buf245, (16, 128, 128), (16384, 128, 1), 0); del buf245  # reuse
        # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf252, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf251, (16, 128, 128), (128, 2048, 1), 0), out=buf253)
        buf254 = reinterpret_tensor(buf252, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf252  # reuse
        # Source Nodes: [tensor_59], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf253, buf254, 262144, grid=grid(262144), stream=stream0)
        buf255 = reinterpret_tensor(buf253, (128, 2048), (2048, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf254, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg150_1, (2048, 2048), (1, 2048), 0), out=buf255)
        del arg150_1
        buf256 = reinterpret_tensor(buf255, (1, 128, 2048), (262144, 2048, 1), 0); del buf255  # reuse
        buf260 = reinterpret_tensor(buf254, (1, 128, 2048), (262144, 2048, 1), 0); del buf254  # reuse
        # Source Nodes: [hidden_states_103, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf256, arg151_1, buf234, buf241, arg144_1, arg152_1, arg153_1, buf260, 128, 2048, grid=grid(128), stream=stream0)
        del arg144_1
        del arg151_1
        del arg152_1
        del arg153_1
        buf261 = reinterpret_tensor(buf240, (128, 8192), (8192, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg154_1, (2048, 8192), (1, 2048), 0), out=buf261)
        del arg154_1
        buf262 = reinterpret_tensor(buf261, (1, 128, 8192), (1048576, 8192, 1), 0); del buf261  # reuse
        # Source Nodes: [add_46, add_47, hidden_states_105, mul_44, mul_45, mul_46, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf262, arg155_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg155_1
        buf263 = reinterpret_tensor(buf260, (128, 2048), (2048, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf262, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg156_1, (8192, 2048), (1, 8192), 0), out=buf263)
        del arg156_1
        buf267 = reinterpret_tensor(buf241, (1, 128, 2048), (262144, 2048, 1), 0); del buf241  # reuse
        # Source Nodes: [hidden_states_110, residual_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf256, buf263, arg157_1, arg158_1, arg159_1, buf267, 128, 2048, grid=grid(128), stream=stream0)
        del arg158_1
        del arg159_1
        buf268 = reinterpret_tensor(buf234, (128, 2048), (2048, 1), 0); del buf234  # reuse
        # Source Nodes: [query_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg160_1, (2048, 2048), (1, 2048), 0), out=buf268)
        del arg160_1
        buf269 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg161_1, (2048, 2048), (1, 2048), 0), out=buf269)
        del arg161_1
        buf270 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf268, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf269, (16, 128, 128), (128, 1, 2048), 0), out=buf270)
        buf274 = reinterpret_tensor(buf268, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf268  # reuse
        # Source Nodes: [attn_weights_73, attn_weights_74, mask_value_12], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg329_1, buf270, buf274, 2048, 128, grid=grid(2048), stream=stream0)
        del arg329_1
        buf273 = reinterpret_tensor(buf270, (128, 2048), (2048, 1), 0); del buf270  # reuse
        # Source Nodes: [value_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg162_1, (2048, 2048), (1, 2048), 0), out=buf273)
        del arg162_1
        buf275 = reinterpret_tensor(buf267, (16, 128, 128), (16384, 128, 1), 0); del buf267  # reuse
        # Source Nodes: [attn_output_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf274, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf273, (16, 128, 128), (128, 2048, 1), 0), out=buf275)
        buf276 = reinterpret_tensor(buf274, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf274  # reuse
        # Source Nodes: [tensor_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf275, buf276, 262144, grid=grid(262144), stream=stream0)
        buf277 = reinterpret_tensor(buf275, (128, 2048), (2048, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg163_1, (2048, 2048), (1, 2048), 0), out=buf277)
        del arg163_1
        buf278 = reinterpret_tensor(buf277, (1, 128, 2048), (262144, 2048, 1), 0); del buf277  # reuse
        buf282 = reinterpret_tensor(buf276, (1, 128, 2048), (262144, 2048, 1), 0); del buf276  # reuse
        # Source Nodes: [hidden_states_112, residual_24, residual_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf278, arg164_1, buf256, buf263, arg157_1, arg165_1, arg166_1, buf282, 128, 2048, grid=grid(128), stream=stream0)
        del arg157_1
        del arg164_1
        del arg165_1
        del arg166_1
        buf283 = reinterpret_tensor(buf262, (128, 8192), (8192, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg167_1, (2048, 8192), (1, 2048), 0), out=buf283)
        del arg167_1
        buf284 = reinterpret_tensor(buf283, (1, 128, 8192), (1048576, 8192, 1), 0); del buf283  # reuse
        # Source Nodes: [add_50, add_51, hidden_states_114, mul_48, mul_49, mul_50, pow_13, tanh_12], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf284, arg168_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg168_1
        buf285 = reinterpret_tensor(buf282, (128, 2048), (2048, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg169_1, (8192, 2048), (1, 8192), 0), out=buf285)
        del arg169_1
        buf289 = reinterpret_tensor(buf263, (1, 128, 2048), (262144, 2048, 1), 0); del buf263  # reuse
        # Source Nodes: [hidden_states_119, residual_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf278, buf285, arg170_1, arg171_1, arg172_1, buf289, 128, 2048, grid=grid(128), stream=stream0)
        del arg171_1
        del arg172_1
        buf290 = reinterpret_tensor(buf256, (128, 2048), (2048, 1), 0); del buf256  # reuse
        # Source Nodes: [query_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg173_1, (2048, 2048), (1, 2048), 0), out=buf290)
        del arg173_1
        buf291 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg174_1, (2048, 2048), (1, 2048), 0), out=buf291)
        del arg174_1
        buf292 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf290, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf291, (16, 128, 128), (128, 1, 2048), 0), out=buf292)
        buf296 = reinterpret_tensor(buf290, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf290  # reuse
        # Source Nodes: [attn_weights_79, attn_weights_80, mask_value_13], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg330_1, buf292, buf296, 2048, 128, grid=grid(2048), stream=stream0)
        del arg330_1
        buf295 = reinterpret_tensor(buf292, (128, 2048), (2048, 1), 0); del buf292  # reuse
        # Source Nodes: [value_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg175_1, (2048, 2048), (1, 2048), 0), out=buf295)
        del arg175_1
        buf297 = reinterpret_tensor(buf289, (16, 128, 128), (16384, 128, 1), 0); del buf289  # reuse
        # Source Nodes: [attn_output_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf295, (16, 128, 128), (128, 2048, 1), 0), out=buf297)
        buf298 = reinterpret_tensor(buf296, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf296  # reuse
        # Source Nodes: [tensor_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf297, buf298, 262144, grid=grid(262144), stream=stream0)
        buf299 = reinterpret_tensor(buf297, (128, 2048), (2048, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg176_1, (2048, 2048), (1, 2048), 0), out=buf299)
        del arg176_1
        buf300 = reinterpret_tensor(buf299, (1, 128, 2048), (262144, 2048, 1), 0); del buf299  # reuse
        buf304 = reinterpret_tensor(buf298, (1, 128, 2048), (262144, 2048, 1), 0); del buf298  # reuse
        # Source Nodes: [hidden_states_121, residual_26, residual_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf300, arg177_1, buf278, buf285, arg170_1, arg178_1, arg179_1, buf304, 128, 2048, grid=grid(128), stream=stream0)
        del arg170_1
        del arg177_1
        del arg178_1
        del arg179_1
        buf305 = reinterpret_tensor(buf284, (128, 8192), (8192, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg180_1, (2048, 8192), (1, 2048), 0), out=buf305)
        del arg180_1
        buf306 = reinterpret_tensor(buf305, (1, 128, 8192), (1048576, 8192, 1), 0); del buf305  # reuse
        # Source Nodes: [add_54, add_55, hidden_states_123, mul_52, mul_53, mul_54, pow_14, tanh_13], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf306, arg181_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg181_1
        buf307 = reinterpret_tensor(buf304, (128, 2048), (2048, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg182_1, (8192, 2048), (1, 8192), 0), out=buf307)
        del arg182_1
        buf311 = reinterpret_tensor(buf285, (1, 128, 2048), (262144, 2048, 1), 0); del buf285  # reuse
        # Source Nodes: [hidden_states_128, residual_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf300, buf307, arg183_1, arg184_1, arg185_1, buf311, 128, 2048, grid=grid(128), stream=stream0)
        del arg184_1
        del arg185_1
        buf312 = reinterpret_tensor(buf278, (128, 2048), (2048, 1), 0); del buf278  # reuse
        # Source Nodes: [query_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg186_1, (2048, 2048), (1, 2048), 0), out=buf312)
        del arg186_1
        buf313 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg187_1, (2048, 2048), (1, 2048), 0), out=buf313)
        del arg187_1
        buf314 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf312, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf313, (16, 128, 128), (128, 1, 2048), 0), out=buf314)
        buf318 = reinterpret_tensor(buf312, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf312  # reuse
        # Source Nodes: [attn_weights_85, attn_weights_86, mask_value_14], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg331_1, buf314, buf318, 2048, 128, grid=grid(2048), stream=stream0)
        del arg331_1
        buf317 = reinterpret_tensor(buf314, (128, 2048), (2048, 1), 0); del buf314  # reuse
        # Source Nodes: [value_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg188_1, (2048, 2048), (1, 2048), 0), out=buf317)
        del arg188_1
        buf319 = reinterpret_tensor(buf311, (16, 128, 128), (16384, 128, 1), 0); del buf311  # reuse
        # Source Nodes: [attn_output_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf318, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf317, (16, 128, 128), (128, 2048, 1), 0), out=buf319)
        buf320 = reinterpret_tensor(buf318, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf318  # reuse
        # Source Nodes: [tensor_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf319, buf320, 262144, grid=grid(262144), stream=stream0)
        buf321 = reinterpret_tensor(buf319, (128, 2048), (2048, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf320, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg189_1, (2048, 2048), (1, 2048), 0), out=buf321)
        del arg189_1
        buf322 = reinterpret_tensor(buf321, (1, 128, 2048), (262144, 2048, 1), 0); del buf321  # reuse
        buf326 = reinterpret_tensor(buf320, (1, 128, 2048), (262144, 2048, 1), 0); del buf320  # reuse
        # Source Nodes: [hidden_states_130, residual_28, residual_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf322, arg190_1, buf300, buf307, arg183_1, arg191_1, arg192_1, buf326, 128, 2048, grid=grid(128), stream=stream0)
        del arg183_1
        del arg190_1
        del arg191_1
        del arg192_1
        buf327 = reinterpret_tensor(buf306, (128, 8192), (8192, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg193_1, (2048, 8192), (1, 2048), 0), out=buf327)
        del arg193_1
        buf328 = reinterpret_tensor(buf327, (1, 128, 8192), (1048576, 8192, 1), 0); del buf327  # reuse
        # Source Nodes: [add_58, add_59, hidden_states_132, mul_56, mul_57, mul_58, pow_15, tanh_14], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf328, arg194_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg194_1
        buf329 = reinterpret_tensor(buf326, (128, 2048), (2048, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg195_1, (8192, 2048), (1, 8192), 0), out=buf329)
        del arg195_1
        buf333 = reinterpret_tensor(buf307, (1, 128, 2048), (262144, 2048, 1), 0); del buf307  # reuse
        # Source Nodes: [hidden_states_137, residual_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf322, buf329, arg196_1, arg197_1, arg198_1, buf333, 128, 2048, grid=grid(128), stream=stream0)
        del arg197_1
        del arg198_1
        buf334 = reinterpret_tensor(buf300, (128, 2048), (2048, 1), 0); del buf300  # reuse
        # Source Nodes: [query_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg199_1, (2048, 2048), (1, 2048), 0), out=buf334)
        del arg199_1
        buf335 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg200_1, (2048, 2048), (1, 2048), 0), out=buf335)
        del arg200_1
        buf336 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf334, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf335, (16, 128, 128), (128, 1, 2048), 0), out=buf336)
        buf340 = reinterpret_tensor(buf334, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf334  # reuse
        # Source Nodes: [attn_weights_91, attn_weights_92, mask_value_15], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg332_1, buf336, buf340, 2048, 128, grid=grid(2048), stream=stream0)
        del arg332_1
        buf339 = reinterpret_tensor(buf336, (128, 2048), (2048, 1), 0); del buf336  # reuse
        # Source Nodes: [value_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg201_1, (2048, 2048), (1, 2048), 0), out=buf339)
        del arg201_1
        buf341 = reinterpret_tensor(buf333, (16, 128, 128), (16384, 128, 1), 0); del buf333  # reuse
        # Source Nodes: [attn_output_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf339, (16, 128, 128), (128, 2048, 1), 0), out=buf341)
        buf342 = reinterpret_tensor(buf340, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf340  # reuse
        # Source Nodes: [tensor_79], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf341, buf342, 262144, grid=grid(262144), stream=stream0)
        buf343 = reinterpret_tensor(buf341, (128, 2048), (2048, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf342, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg202_1, (2048, 2048), (1, 2048), 0), out=buf343)
        del arg202_1
        buf344 = reinterpret_tensor(buf343, (1, 128, 2048), (262144, 2048, 1), 0); del buf343  # reuse
        buf348 = reinterpret_tensor(buf342, (1, 128, 2048), (262144, 2048, 1), 0); del buf342  # reuse
        # Source Nodes: [hidden_states_139, residual_30, residual_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf344, arg203_1, buf322, buf329, arg196_1, arg204_1, arg205_1, buf348, 128, 2048, grid=grid(128), stream=stream0)
        del arg196_1
        del arg203_1
        del arg204_1
        del arg205_1
        buf349 = reinterpret_tensor(buf328, (128, 8192), (8192, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf348, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg206_1, (2048, 8192), (1, 2048), 0), out=buf349)
        del arg206_1
        buf350 = reinterpret_tensor(buf349, (1, 128, 8192), (1048576, 8192, 1), 0); del buf349  # reuse
        # Source Nodes: [add_62, add_63, hidden_states_141, mul_60, mul_61, mul_62, pow_16, tanh_15], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf350, arg207_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg207_1
        buf351 = reinterpret_tensor(buf348, (128, 2048), (2048, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg208_1, (8192, 2048), (1, 8192), 0), out=buf351)
        del arg208_1
        buf355 = reinterpret_tensor(buf329, (1, 128, 2048), (262144, 2048, 1), 0); del buf329  # reuse
        # Source Nodes: [hidden_states_146, residual_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf344, buf351, arg209_1, arg210_1, arg211_1, buf355, 128, 2048, grid=grid(128), stream=stream0)
        del arg210_1
        del arg211_1
        buf356 = reinterpret_tensor(buf322, (128, 2048), (2048, 1), 0); del buf322  # reuse
        # Source Nodes: [query_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg212_1, (2048, 2048), (1, 2048), 0), out=buf356)
        del arg212_1
        buf357 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg213_1, (2048, 2048), (1, 2048), 0), out=buf357)
        del arg213_1
        buf358 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_96], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf356, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf357, (16, 128, 128), (128, 1, 2048), 0), out=buf358)
        buf362 = reinterpret_tensor(buf356, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf356  # reuse
        # Source Nodes: [attn_weights_97, attn_weights_98, mask_value_16], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg333_1, buf358, buf362, 2048, 128, grid=grid(2048), stream=stream0)
        del arg333_1
        buf361 = reinterpret_tensor(buf358, (128, 2048), (2048, 1), 0); del buf358  # reuse
        # Source Nodes: [value_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg214_1, (2048, 2048), (1, 2048), 0), out=buf361)
        del arg214_1
        buf363 = reinterpret_tensor(buf355, (16, 128, 128), (16384, 128, 1), 0); del buf355  # reuse
        # Source Nodes: [attn_output_96], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf361, (16, 128, 128), (128, 2048, 1), 0), out=buf363)
        buf364 = reinterpret_tensor(buf362, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf362  # reuse
        # Source Nodes: [tensor_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf363, buf364, 262144, grid=grid(262144), stream=stream0)
        buf365 = reinterpret_tensor(buf363, (128, 2048), (2048, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg215_1, (2048, 2048), (1, 2048), 0), out=buf365)
        del arg215_1
        buf366 = reinterpret_tensor(buf365, (1, 128, 2048), (262144, 2048, 1), 0); del buf365  # reuse
        buf370 = reinterpret_tensor(buf364, (1, 128, 2048), (262144, 2048, 1), 0); del buf364  # reuse
        # Source Nodes: [hidden_states_148, residual_32, residual_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf366, arg216_1, buf344, buf351, arg209_1, arg217_1, arg218_1, buf370, 128, 2048, grid=grid(128), stream=stream0)
        del arg209_1
        del arg216_1
        del arg217_1
        del arg218_1
        buf371 = reinterpret_tensor(buf350, (128, 8192), (8192, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg219_1, (2048, 8192), (1, 2048), 0), out=buf371)
        del arg219_1
        buf372 = reinterpret_tensor(buf371, (1, 128, 8192), (1048576, 8192, 1), 0); del buf371  # reuse
        # Source Nodes: [add_66, add_67, hidden_states_150, mul_64, mul_65, mul_66, pow_17, tanh_16], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf372, arg220_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg220_1
        buf373 = reinterpret_tensor(buf370, (128, 2048), (2048, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg221_1, (8192, 2048), (1, 8192), 0), out=buf373)
        del arg221_1
        buf377 = reinterpret_tensor(buf351, (1, 128, 2048), (262144, 2048, 1), 0); del buf351  # reuse
        # Source Nodes: [hidden_states_155, residual_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf366, buf373, arg222_1, arg223_1, arg224_1, buf377, 128, 2048, grid=grid(128), stream=stream0)
        del arg223_1
        del arg224_1
        buf378 = reinterpret_tensor(buf344, (128, 2048), (2048, 1), 0); del buf344  # reuse
        # Source Nodes: [query_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg225_1, (2048, 2048), (1, 2048), 0), out=buf378)
        del arg225_1
        buf379 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg226_1, (2048, 2048), (1, 2048), 0), out=buf379)
        del arg226_1
        buf380 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_102], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf378, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf379, (16, 128, 128), (128, 1, 2048), 0), out=buf380)
        buf384 = reinterpret_tensor(buf378, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf378  # reuse
        # Source Nodes: [attn_weights_103, attn_weights_104, mask_value_17], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg334_1, buf380, buf384, 2048, 128, grid=grid(2048), stream=stream0)
        del arg334_1
        buf383 = reinterpret_tensor(buf380, (128, 2048), (2048, 1), 0); del buf380  # reuse
        # Source Nodes: [value_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg227_1, (2048, 2048), (1, 2048), 0), out=buf383)
        del arg227_1
        buf385 = reinterpret_tensor(buf377, (16, 128, 128), (16384, 128, 1), 0); del buf377  # reuse
        # Source Nodes: [attn_output_102], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf384, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf383, (16, 128, 128), (128, 2048, 1), 0), out=buf385)
        buf386 = reinterpret_tensor(buf384, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf384  # reuse
        # Source Nodes: [tensor_89], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf385, buf386, 262144, grid=grid(262144), stream=stream0)
        buf387 = reinterpret_tensor(buf385, (128, 2048), (2048, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf386, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg228_1, (2048, 2048), (1, 2048), 0), out=buf387)
        del arg228_1
        buf388 = reinterpret_tensor(buf387, (1, 128, 2048), (262144, 2048, 1), 0); del buf387  # reuse
        buf392 = reinterpret_tensor(buf386, (1, 128, 2048), (262144, 2048, 1), 0); del buf386  # reuse
        # Source Nodes: [hidden_states_157, residual_34, residual_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf388, arg229_1, buf366, buf373, arg222_1, arg230_1, arg231_1, buf392, 128, 2048, grid=grid(128), stream=stream0)
        del arg222_1
        del arg229_1
        del arg230_1
        del arg231_1
        buf393 = reinterpret_tensor(buf372, (128, 8192), (8192, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg232_1, (2048, 8192), (1, 2048), 0), out=buf393)
        del arg232_1
        buf394 = reinterpret_tensor(buf393, (1, 128, 8192), (1048576, 8192, 1), 0); del buf393  # reuse
        # Source Nodes: [add_70, add_71, hidden_states_159, mul_68, mul_69, mul_70, pow_18, tanh_17], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf394, arg233_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg233_1
        buf395 = reinterpret_tensor(buf392, (128, 2048), (2048, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg234_1, (8192, 2048), (1, 8192), 0), out=buf395)
        del arg234_1
        buf399 = reinterpret_tensor(buf373, (1, 128, 2048), (262144, 2048, 1), 0); del buf373  # reuse
        # Source Nodes: [hidden_states_164, residual_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf388, buf395, arg235_1, arg236_1, arg237_1, buf399, 128, 2048, grid=grid(128), stream=stream0)
        del arg236_1
        del arg237_1
        buf400 = reinterpret_tensor(buf366, (128, 2048), (2048, 1), 0); del buf366  # reuse
        # Source Nodes: [query_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg238_1, (2048, 2048), (1, 2048), 0), out=buf400)
        del arg238_1
        buf401 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg239_1, (2048, 2048), (1, 2048), 0), out=buf401)
        del arg239_1
        buf402 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_108], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf400, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf401, (16, 128, 128), (128, 1, 2048), 0), out=buf402)
        buf406 = reinterpret_tensor(buf400, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf400  # reuse
        # Source Nodes: [attn_weights_109, attn_weights_110, mask_value_18], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg335_1, buf402, buf406, 2048, 128, grid=grid(2048), stream=stream0)
        del arg335_1
        buf405 = reinterpret_tensor(buf402, (128, 2048), (2048, 1), 0); del buf402  # reuse
        # Source Nodes: [value_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg240_1, (2048, 2048), (1, 2048), 0), out=buf405)
        del arg240_1
        buf407 = reinterpret_tensor(buf399, (16, 128, 128), (16384, 128, 1), 0); del buf399  # reuse
        # Source Nodes: [attn_output_108], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf406, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf405, (16, 128, 128), (128, 2048, 1), 0), out=buf407)
        buf408 = reinterpret_tensor(buf406, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf406  # reuse
        # Source Nodes: [tensor_94], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf407, buf408, 262144, grid=grid(262144), stream=stream0)
        buf409 = reinterpret_tensor(buf407, (128, 2048), (2048, 1), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf408, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg241_1, (2048, 2048), (1, 2048), 0), out=buf409)
        del arg241_1
        buf410 = reinterpret_tensor(buf409, (1, 128, 2048), (262144, 2048, 1), 0); del buf409  # reuse
        buf414 = reinterpret_tensor(buf408, (1, 128, 2048), (262144, 2048, 1), 0); del buf408  # reuse
        # Source Nodes: [hidden_states_166, residual_36, residual_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf410, arg242_1, buf388, buf395, arg235_1, arg243_1, arg244_1, buf414, 128, 2048, grid=grid(128), stream=stream0)
        del arg235_1
        del arg242_1
        del arg243_1
        del arg244_1
        buf415 = reinterpret_tensor(buf394, (128, 8192), (8192, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf414, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg245_1, (2048, 8192), (1, 2048), 0), out=buf415)
        del arg245_1
        buf416 = reinterpret_tensor(buf415, (1, 128, 8192), (1048576, 8192, 1), 0); del buf415  # reuse
        # Source Nodes: [add_74, add_75, hidden_states_168, mul_72, mul_73, mul_74, pow_19, tanh_18], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf416, arg246_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg246_1
        buf417 = reinterpret_tensor(buf414, (128, 2048), (2048, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg247_1, (8192, 2048), (1, 8192), 0), out=buf417)
        del arg247_1
        buf421 = reinterpret_tensor(buf395, (1, 128, 2048), (262144, 2048, 1), 0); del buf395  # reuse
        # Source Nodes: [hidden_states_173, residual_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf410, buf417, arg248_1, arg249_1, arg250_1, buf421, 128, 2048, grid=grid(128), stream=stream0)
        del arg249_1
        del arg250_1
        buf422 = reinterpret_tensor(buf388, (128, 2048), (2048, 1), 0); del buf388  # reuse
        # Source Nodes: [query_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg251_1, (2048, 2048), (1, 2048), 0), out=buf422)
        del arg251_1
        buf423 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg252_1, (2048, 2048), (1, 2048), 0), out=buf423)
        del arg252_1
        buf424 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_114], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf422, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf423, (16, 128, 128), (128, 1, 2048), 0), out=buf424)
        buf428 = reinterpret_tensor(buf422, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf422  # reuse
        # Source Nodes: [attn_weights_115, attn_weights_116, mask_value_19], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg336_1, buf424, buf428, 2048, 128, grid=grid(2048), stream=stream0)
        del arg336_1
        buf427 = reinterpret_tensor(buf424, (128, 2048), (2048, 1), 0); del buf424  # reuse
        # Source Nodes: [value_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg253_1, (2048, 2048), (1, 2048), 0), out=buf427)
        del arg253_1
        buf429 = reinterpret_tensor(buf421, (16, 128, 128), (16384, 128, 1), 0); del buf421  # reuse
        # Source Nodes: [attn_output_114], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf428, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf427, (16, 128, 128), (128, 2048, 1), 0), out=buf429)
        buf430 = reinterpret_tensor(buf428, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf428  # reuse
        # Source Nodes: [tensor_99], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf429, buf430, 262144, grid=grid(262144), stream=stream0)
        buf431 = reinterpret_tensor(buf429, (128, 2048), (2048, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf430, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg254_1, (2048, 2048), (1, 2048), 0), out=buf431)
        del arg254_1
        buf432 = reinterpret_tensor(buf431, (1, 128, 2048), (262144, 2048, 1), 0); del buf431  # reuse
        buf436 = reinterpret_tensor(buf430, (1, 128, 2048), (262144, 2048, 1), 0); del buf430  # reuse
        # Source Nodes: [hidden_states_175, residual_38, residual_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf432, arg255_1, buf410, buf417, arg248_1, arg256_1, arg257_1, buf436, 128, 2048, grid=grid(128), stream=stream0)
        del arg248_1
        del arg255_1
        del arg256_1
        del arg257_1
        buf437 = reinterpret_tensor(buf416, (128, 8192), (8192, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg258_1, (2048, 8192), (1, 2048), 0), out=buf437)
        del arg258_1
        buf438 = reinterpret_tensor(buf437, (1, 128, 8192), (1048576, 8192, 1), 0); del buf437  # reuse
        # Source Nodes: [add_78, add_79, hidden_states_177, mul_76, mul_77, mul_78, pow_20, tanh_19], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf438, arg259_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg259_1
        buf439 = reinterpret_tensor(buf436, (128, 2048), (2048, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf438, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg260_1, (8192, 2048), (1, 8192), 0), out=buf439)
        del arg260_1
        buf443 = reinterpret_tensor(buf417, (1, 128, 2048), (262144, 2048, 1), 0); del buf417  # reuse
        # Source Nodes: [hidden_states_182, residual_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf432, buf439, arg261_1, arg262_1, arg263_1, buf443, 128, 2048, grid=grid(128), stream=stream0)
        del arg262_1
        del arg263_1
        buf444 = reinterpret_tensor(buf410, (128, 2048), (2048, 1), 0); del buf410  # reuse
        # Source Nodes: [query_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg264_1, (2048, 2048), (1, 2048), 0), out=buf444)
        del arg264_1
        buf445 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg265_1, (2048, 2048), (1, 2048), 0), out=buf445)
        del arg265_1
        buf446 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_120], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf444, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf445, (16, 128, 128), (128, 1, 2048), 0), out=buf446)
        buf450 = reinterpret_tensor(buf444, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf444  # reuse
        # Source Nodes: [attn_weights_121, attn_weights_122, mask_value_20], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg337_1, buf446, buf450, 2048, 128, grid=grid(2048), stream=stream0)
        del arg337_1
        buf449 = reinterpret_tensor(buf446, (128, 2048), (2048, 1), 0); del buf446  # reuse
        # Source Nodes: [value_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg266_1, (2048, 2048), (1, 2048), 0), out=buf449)
        del arg266_1
        buf451 = reinterpret_tensor(buf443, (16, 128, 128), (16384, 128, 1), 0); del buf443  # reuse
        # Source Nodes: [attn_output_120], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf449, (16, 128, 128), (128, 2048, 1), 0), out=buf451)
        buf452 = reinterpret_tensor(buf450, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf450  # reuse
        # Source Nodes: [tensor_104], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf451, buf452, 262144, grid=grid(262144), stream=stream0)
        buf453 = reinterpret_tensor(buf451, (128, 2048), (2048, 1), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf452, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg267_1, (2048, 2048), (1, 2048), 0), out=buf453)
        del arg267_1
        buf454 = reinterpret_tensor(buf453, (1, 128, 2048), (262144, 2048, 1), 0); del buf453  # reuse
        buf458 = reinterpret_tensor(buf452, (1, 128, 2048), (262144, 2048, 1), 0); del buf452  # reuse
        # Source Nodes: [hidden_states_184, residual_40, residual_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf454, arg268_1, buf432, buf439, arg261_1, arg269_1, arg270_1, buf458, 128, 2048, grid=grid(128), stream=stream0)
        del arg261_1
        del arg268_1
        del arg269_1
        del arg270_1
        buf459 = reinterpret_tensor(buf438, (128, 8192), (8192, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg271_1, (2048, 8192), (1, 2048), 0), out=buf459)
        del arg271_1
        buf460 = reinterpret_tensor(buf459, (1, 128, 8192), (1048576, 8192, 1), 0); del buf459  # reuse
        # Source Nodes: [add_82, add_83, hidden_states_186, mul_80, mul_81, mul_82, pow_21, tanh_20], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf460, arg272_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg272_1
        buf461 = reinterpret_tensor(buf458, (128, 2048), (2048, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg273_1, (8192, 2048), (1, 8192), 0), out=buf461)
        del arg273_1
        buf465 = reinterpret_tensor(buf439, (1, 128, 2048), (262144, 2048, 1), 0); del buf439  # reuse
        # Source Nodes: [hidden_states_191, residual_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf454, buf461, arg274_1, arg275_1, arg276_1, buf465, 128, 2048, grid=grid(128), stream=stream0)
        del arg275_1
        del arg276_1
        buf466 = reinterpret_tensor(buf432, (128, 2048), (2048, 1), 0); del buf432  # reuse
        # Source Nodes: [query_63], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg277_1, (2048, 2048), (1, 2048), 0), out=buf466)
        del arg277_1
        buf467 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_63], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg278_1, (2048, 2048), (1, 2048), 0), out=buf467)
        del arg278_1
        buf468 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_126], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf466, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf467, (16, 128, 128), (128, 1, 2048), 0), out=buf468)
        buf472 = reinterpret_tensor(buf466, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf466  # reuse
        # Source Nodes: [attn_weights_127, attn_weights_128, mask_value_21], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg338_1, buf468, buf472, 2048, 128, grid=grid(2048), stream=stream0)
        del arg338_1
        buf471 = reinterpret_tensor(buf468, (128, 2048), (2048, 1), 0); del buf468  # reuse
        # Source Nodes: [value_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg279_1, (2048, 2048), (1, 2048), 0), out=buf471)
        del arg279_1
        buf473 = reinterpret_tensor(buf465, (16, 128, 128), (16384, 128, 1), 0); del buf465  # reuse
        # Source Nodes: [attn_output_126], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf472, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf471, (16, 128, 128), (128, 2048, 1), 0), out=buf473)
        buf474 = reinterpret_tensor(buf472, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf472  # reuse
        # Source Nodes: [tensor_109], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf473, buf474, 262144, grid=grid(262144), stream=stream0)
        buf475 = reinterpret_tensor(buf473, (128, 2048), (2048, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg280_1, (2048, 2048), (1, 2048), 0), out=buf475)
        del arg280_1
        buf476 = reinterpret_tensor(buf475, (1, 128, 2048), (262144, 2048, 1), 0); del buf475  # reuse
        buf480 = reinterpret_tensor(buf474, (1, 128, 2048), (262144, 2048, 1), 0); del buf474  # reuse
        # Source Nodes: [hidden_states_193, residual_42, residual_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf476, arg281_1, buf454, buf461, arg274_1, arg282_1, arg283_1, buf480, 128, 2048, grid=grid(128), stream=stream0)
        del arg274_1
        del arg281_1
        del arg282_1
        del arg283_1
        buf481 = reinterpret_tensor(buf460, (128, 8192), (8192, 1), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf480, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg284_1, (2048, 8192), (1, 2048), 0), out=buf481)
        del arg284_1
        buf482 = reinterpret_tensor(buf481, (1, 128, 8192), (1048576, 8192, 1), 0); del buf481  # reuse
        # Source Nodes: [add_86, add_87, hidden_states_195, mul_84, mul_85, mul_86, pow_22, tanh_21], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf482, arg285_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg285_1
        buf483 = reinterpret_tensor(buf480, (128, 2048), (2048, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf482, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg286_1, (8192, 2048), (1, 8192), 0), out=buf483)
        del arg286_1
        buf487 = reinterpret_tensor(buf461, (1, 128, 2048), (262144, 2048, 1), 0); del buf461  # reuse
        # Source Nodes: [hidden_states_200, residual_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf476, buf483, arg287_1, arg288_1, arg289_1, buf487, 128, 2048, grid=grid(128), stream=stream0)
        del arg288_1
        del arg289_1
        buf488 = reinterpret_tensor(buf454, (128, 2048), (2048, 1), 0); del buf454  # reuse
        # Source Nodes: [query_66], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg290_1, (2048, 2048), (1, 2048), 0), out=buf488)
        del arg290_1
        buf489 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_66], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg291_1, (2048, 2048), (1, 2048), 0), out=buf489)
        del arg291_1
        buf490 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_132], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf488, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf489, (16, 128, 128), (128, 1, 2048), 0), out=buf490)
        buf494 = reinterpret_tensor(buf488, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf488  # reuse
        # Source Nodes: [attn_weights_133, attn_weights_134, mask_value_22], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg339_1, buf490, buf494, 2048, 128, grid=grid(2048), stream=stream0)
        del arg339_1
        buf493 = reinterpret_tensor(buf490, (128, 2048), (2048, 1), 0); del buf490  # reuse
        # Source Nodes: [value_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg292_1, (2048, 2048), (1, 2048), 0), out=buf493)
        del arg292_1
        buf495 = reinterpret_tensor(buf487, (16, 128, 128), (16384, 128, 1), 0); del buf487  # reuse
        # Source Nodes: [attn_output_132], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf494, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf493, (16, 128, 128), (128, 2048, 1), 0), out=buf495)
        buf496 = reinterpret_tensor(buf494, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf494  # reuse
        # Source Nodes: [tensor_114], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf495, buf496, 262144, grid=grid(262144), stream=stream0)
        buf497 = reinterpret_tensor(buf495, (128, 2048), (2048, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf496, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg293_1, (2048, 2048), (1, 2048), 0), out=buf497)
        del arg293_1
        buf498 = reinterpret_tensor(buf497, (1, 128, 2048), (262144, 2048, 1), 0); del buf497  # reuse
        buf502 = reinterpret_tensor(buf496, (1, 128, 2048), (262144, 2048, 1), 0); del buf496  # reuse
        # Source Nodes: [hidden_states_202, residual_44, residual_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf498, arg294_1, buf476, buf483, arg287_1, arg295_1, arg296_1, buf502, 128, 2048, grid=grid(128), stream=stream0)
        del arg287_1
        del arg294_1
        del arg295_1
        del arg296_1
        buf503 = reinterpret_tensor(buf482, (128, 8192), (8192, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf502, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg297_1, (2048, 8192), (1, 2048), 0), out=buf503)
        del arg297_1
        buf504 = reinterpret_tensor(buf503, (1, 128, 8192), (1048576, 8192, 1), 0); del buf503  # reuse
        # Source Nodes: [add_90, add_91, hidden_states_204, mul_88, mul_89, mul_90, pow_23, tanh_22], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf504, arg298_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg298_1
        buf505 = reinterpret_tensor(buf502, (128, 2048), (2048, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf504, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg299_1, (8192, 2048), (1, 8192), 0), out=buf505)
        del arg299_1
        buf509 = reinterpret_tensor(buf483, (1, 128, 2048), (262144, 2048, 1), 0); del buf483  # reuse
        # Source Nodes: [hidden_states_209, residual_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf498, buf505, arg300_1, arg301_1, arg302_1, buf509, 128, 2048, grid=grid(128), stream=stream0)
        del arg301_1
        del arg302_1
        buf510 = reinterpret_tensor(buf476, (128, 2048), (2048, 1), 0); del buf476  # reuse
        # Source Nodes: [query_69], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg303_1, (2048, 2048), (1, 2048), 0), out=buf510)
        del arg303_1
        buf511 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_69], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg304_1, (2048, 2048), (1, 2048), 0), out=buf511)
        del arg304_1
        buf512 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_138], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf510, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf511, (16, 128, 128), (128, 1, 2048), 0), out=buf512)
        buf516 = reinterpret_tensor(buf510, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf510  # reuse
        # Source Nodes: [attn_weights_139, attn_weights_140, mask_value_23], Original ATen: [aten._softmax, aten._to_copy, aten.where]
        triton_per_fused__softmax__to_copy_where_1.run(arg340_1, buf512, buf516, 2048, 128, grid=grid(2048), stream=stream0)
        del arg340_1
        buf515 = reinterpret_tensor(buf512, (128, 2048), (2048, 1), 0); del buf512  # reuse
        # Source Nodes: [value_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg305_1, (2048, 2048), (1, 2048), 0), out=buf515)
        del arg305_1
        buf517 = reinterpret_tensor(buf509, (16, 128, 128), (16384, 128, 1), 0); del buf509  # reuse
        # Source Nodes: [attn_output_138], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf516, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf515, (16, 128, 128), (128, 2048, 1), 0), out=buf517)
        buf518 = reinterpret_tensor(buf516, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf516  # reuse
        # Source Nodes: [tensor_119], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf517, buf518, 262144, grid=grid(262144), stream=stream0)
        buf519 = reinterpret_tensor(buf517, (128, 2048), (2048, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf518, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg306_1, (2048, 2048), (1, 2048), 0), out=buf519)
        del arg306_1
        buf520 = reinterpret_tensor(buf519, (1, 128, 2048), (262144, 2048, 1), 0); del buf519  # reuse
        buf524 = reinterpret_tensor(buf518, (1, 128, 2048), (262144, 2048, 1), 0); del buf518  # reuse
        # Source Nodes: [hidden_states_211, residual_46, residual_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf520, arg307_1, buf498, buf505, arg300_1, arg308_1, arg309_1, buf524, 128, 2048, grid=grid(128), stream=stream0)
        del arg300_1
        del arg307_1
        del arg308_1
        del arg309_1
        del buf498
        buf525 = reinterpret_tensor(buf504, (128, 8192), (8192, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg310_1, (2048, 8192), (1, 2048), 0), out=buf525)
        del arg310_1
        buf526 = reinterpret_tensor(buf525, (1, 128, 8192), (1048576, 8192, 1), 0); del buf525  # reuse
        # Source Nodes: [add_94, add_95, hidden_states_213, mul_92, mul_93, mul_94, pow_24, tanh_23], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_4.run(buf526, arg311_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg311_1
        buf527 = reinterpret_tensor(buf524, (128, 2048), (2048, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf526, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg312_1, (8192, 2048), (1, 8192), 0), out=buf527)
        del arg312_1
        del buf526
        buf531 = reinterpret_tensor(buf505, (1, 128, 2048), (262144, 2048, 1), 0); del buf505  # reuse
        # Source Nodes: [hidden_states_217, hidden_states_218, hidden_states_220], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_red_fused_add_native_layer_norm_5.run(buf520, buf527, arg313_1, arg314_1, arg315_1, buf531, 128, 2048, grid=grid(128), stream=stream0)
        del arg313_1
        del arg314_1
        del arg315_1
        del buf520
        del buf527
        buf532 = empty((128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg316_1, (2048, 2), (1, 2048), 0), out=buf532)
        del arg316_1
        buf533 = empty((1, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [argmax, eq, long], Original ATen: [aten._to_copy, aten.argmax, aten.eq]
        triton_red_fused__to_copy_argmax_eq_7.run(arg341_1, buf533, 1, 128, grid=grid(1), stream=stream0)
        del arg341_1
        buf534 = empty((1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [pooled_logits], Original ATen: [aten.index]
        triton_poi_fused_index_8.run(buf533, buf532, buf534, 2, grid=grid(2), stream=stream0)
        return (buf531, reinterpret_tensor(buf5, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf9, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf27, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf31, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf49, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf53, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf71, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf75, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf93, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf97, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf115, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf119, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf137, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf141, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf159, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf163, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf181, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf185, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf203, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf207, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf225, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf229, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf247, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf251, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf269, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf273, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf291, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf295, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf313, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf317, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf335, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf339, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf357, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf361, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf379, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf383, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf401, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf405, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf423, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf427, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf445, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf449, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf467, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf471, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf489, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf493, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf511, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf515, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), buf534, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((50257, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((2, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg318_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg319_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg320_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg321_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg322_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg323_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg324_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg325_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg326_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg327_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg328_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg329_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg330_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg331_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg332_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg333_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg334_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg335_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg336_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg337_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg338_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg339_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg340_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    arg341_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPTNeoForSequenceClassification', benchmark_compiled_module)
