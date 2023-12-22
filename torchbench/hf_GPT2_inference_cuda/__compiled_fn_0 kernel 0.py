
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


# kernel path: /tmp/torchinductor_youkaichao/mj/cmjdebwsx5mgjnfjoemczkoowseneiqh7gid7cmrj7pyurycyf54.py
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 1024
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50257
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 50257), "index out of bounds: 0 <= tmp3 < 50257")
        tmp4 = tl.load(in_ptr1 + (r2 + (768*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 + 50257
        tmp12 = tmp0 < 0
        tmp13 = tl.where(tmp12, tmp11, tmp0)
        tl.device_assert((0 <= tmp13) & (tmp13 < 50257), "index out of bounds: 0 <= tmp13 < 50257")
        tmp14 = tl.load(in_ptr1 + (r2 + (768*tmp13)), rmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp27, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/wk/cwkefm3rh3tyry5l4o6zl4j6ozgd64ia5pcav6szl7gt5ehwgynx.py
# Source Nodes: [attn_weights], Original ATen: [aten.clone]
# attn_weights => clone_1
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 12
    x3 = (xindex // 786432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (2304*x1) + (2359296*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chfraletlcwrsikpk4avdsvmg4sb66mxewywqmejyc46ecrlioby.py
# Source Nodes: [attn_weights], Original ATen: [aten.clone]
# attn_weights => clone_2
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (768 + y0 + (2304*x2) + (2359296*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pt/cptvghnxvfc5vx5agotm7i4ty5ncy7boqq2ptqhav6xvnk34ec2q.py
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_div_full_where_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 24576
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


# kernel path: /tmp/torchinductor_youkaichao/ua/cuah2372jssewzxtnmj3yjhm7wmc5tpc2pr2oi4h4h57idoizcqw.py
# Source Nodes: [attn_output], Original ATen: [aten.clone]
# attn_output => clone_4
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 12
    x3 = (xindex // 786432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + (64*x2) + (2304*x1) + (2359296*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ik/cikzxzr5wvafv7cx27wq24vlaavhec6j6j3rmno4erjaow42cuy7.py
# Source Nodes: [tensor_3], Original ATen: [aten.clone]
# tensor_3 => clone_5
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768) % 1024
    x3 = (xindex // 786432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1) + (786432*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfwc5d7lsphtntkposxcsd7wgvb4xkxrz5sm7nqhj7y66jwuyyx.py
# Source Nodes: [add, hidden_states_2, inputs_embeds, position_embeds, residual_1], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
# add => add
# hidden_states_2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# inputs_embeds => embedding
# position_embeds => embedding_1
# residual_1 => add_3
triton_per_fused_add_embedding_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + 50257
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert((0 <= tmp6) & (tmp6 < 50257), "index out of bounds: 0 <= tmp6 < 50257")
    tmp7 = tl.load(in_ptr2 + (r2 + (768*tmp6)), rmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
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
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp10, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2p4spxe53lhigywpmwhxxs636vj6ahvvxsvappqafg6o3kp5s5.py
# Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_2 => add_6
# add_3 => add_7
# hidden_states_4 => mul_7
# mul => mul_4
# mul_1 => mul_5
# mul_2 => mul_6
# pow_1 => pow_1
# tanh => tanh
triton_poi_fused_add_mul_pow_tanh_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
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


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hkjzk5ue24tw56zqsti7i2ozrtqgfrwjco24dgfrbfshzzttvy.py
# Source Nodes: [hidden_states_8, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_8 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
# residual_2 => add_8
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rulho3uj5omkk6bdlrtz4qtdpbzm5ugbb4oi4kgildex6zkytp.py
# Source Nodes: [hidden_states_10, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_10 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
# residual_2 => add_8
# residual_3 => add_11
triton_per_fused_add_native_layer_norm_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mu/cmu7kcuszxk7fexcznalzepisoauacq5e5gpiltlzjvz4vb6gsxu.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38599680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 50257, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 50260, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = 0.0
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1 = args
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
    assert_size_stride(arg48_1, (2304, ), (1, ))
    assert_size_stride(arg49_1, (768, 2304), (2304, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, 768), (768, 1))
    assert_size_stride(arg52_1, (3072, ), (1, ))
    assert_size_stride(arg53_1, (768, 3072), (3072, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (3072, 768), (768, 1))
    assert_size_stride(arg56_1, (2304, ), (1, ))
    assert_size_stride(arg57_1, (768, 2304), (2304, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (3072, ), (1, ))
    assert_size_stride(arg61_1, (768, 3072), (3072, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (3072, 768), (768, 1))
    assert_size_stride(arg64_1, (2304, ), (1, ))
    assert_size_stride(arg65_1, (768, 2304), (2304, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, 768), (768, 1))
    assert_size_stride(arg68_1, (3072, ), (1, ))
    assert_size_stride(arg69_1, (768, 3072), (3072, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (3072, 768), (768, 1))
    assert_size_stride(arg72_1, (2304, ), (1, ))
    assert_size_stride(arg73_1, (768, 2304), (2304, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (3072, ), (1, ))
    assert_size_stride(arg77_1, (768, 3072), (3072, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (2304, ), (1, ))
    assert_size_stride(arg81_1, (768, 2304), (2304, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, 768), (768, 1))
    assert_size_stride(arg84_1, (3072, ), (1, ))
    assert_size_stride(arg85_1, (768, 3072), (3072, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (3072, 768), (768, 1))
    assert_size_stride(arg88_1, (2304, ), (1, ))
    assert_size_stride(arg89_1, (768, 2304), (2304, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (3072, ), (1, ))
    assert_size_stride(arg93_1, (768, 3072), (3072, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, 768), (768, 1))
    assert_size_stride(arg96_1, (50257, 768), (768, 1))
    assert_size_stride(arg97_1, (1024, 768), (768, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (50257, 768), (768, 1))
    assert_size_stride(arg149_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg150_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg151_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg152_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg153_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg154_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg155_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg156_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg157_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg158_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg159_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg160_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg161_1, (2, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((2, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, hidden_states, inputs_embeds, position_embeds], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_native_layer_norm_0.run(arg161_1, arg96_1, arg97_1, arg98_1, arg99_1, buf3, 2048, 768, grid=grid(2048), stream=stream0)
        del arg98_1
        del arg99_1
        buf4 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg0_1, reinterpret_tensor(buf3, (2048, 768), (768, 1), 0), arg1_1, alpha=1, beta=1, out=buf4)
        del arg0_1
        del arg1_1
        buf5 = reinterpret_tensor(buf3, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf3  # reuse
        # Source Nodes: [attn_weights], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf4, buf5, 1572864, grid=grid(1572864), stream=stream0)
        buf6 = empty((2, 12, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf4, buf6, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf7 = empty((24, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf5, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf6, (24, 64, 1024), (65536, 1024, 1), 0), out=buf7)
        buf10 = empty((2, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1, attn_weights_2, attn_weights_3, full, mask_value], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg149_1, buf7, buf10, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg149_1
        buf11 = reinterpret_tensor(buf6, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf4, buf11, 1572864, grid=grid(1572864), stream=stream0)
        buf12 = reinterpret_tensor(buf5, (24, 1024, 64), (65536, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf11, (24, 1024, 64), (65536, 64, 1), 0), out=buf12)
        buf13 = reinterpret_tensor(buf11, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [tensor_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf12, buf13, 1572864, grid=grid(1572864), stream=stream0)
        buf14 = reinterpret_tensor(buf12, (2048, 768), (768, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (2048, 768), (768, 1), 0), arg3_1, out=buf14)
        del arg3_1
        buf15 = reinterpret_tensor(buf14, (2, 1024, 768), (786432, 768, 1), 0); del buf14  # reuse
        buf19 = reinterpret_tensor(buf13, (2, 1024, 768), (786432, 768, 1), 0); del buf13  # reuse
        # Source Nodes: [add, hidden_states_2, inputs_embeds, position_embeds, residual_1], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
        triton_per_fused_add_embedding_native_layer_norm_6.run(buf15, arg2_1, arg161_1, arg96_1, arg97_1, arg100_1, arg101_1, buf19, 2048, 768, grid=grid(2048), stream=stream0)
        del arg100_1
        del arg101_1
        del arg161_1
        del arg2_1
        del arg96_1
        del arg97_1
        buf20 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (2048, 768), (768, 1), 0), arg5_1, out=buf20)
        del arg5_1
        buf21 = reinterpret_tensor(buf20, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf20  # reuse
        # Source Nodes: [add_2, add_3, hidden_states_4, mul, mul_1, mul_2, pow_1, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf21, arg4_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg4_1
        buf22 = reinterpret_tensor(buf19, (2048, 768), (768, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (2048, 3072), (3072, 1), 0), arg7_1, out=buf22)
        del arg7_1
        buf26 = empty((2, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8, residual_2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf15, buf22, arg6_1, arg102_1, arg103_1, buf26, 2048, 768, grid=grid(2048), stream=stream0)
        del arg102_1
        del arg103_1
        buf27 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf26, (2048, 768), (768, 1), 0), arg9_1, alpha=1, beta=1, out=buf27)
        del arg8_1
        del arg9_1
        buf28 = reinterpret_tensor(buf26, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf26  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf27, buf28, 1572864, grid=grid(1572864), stream=stream0)
        buf29 = empty((2, 12, 64, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf27, buf29, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf30 = reinterpret_tensor(buf10, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf10  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf29, (24, 64, 1024), (65536, 1024, 1), 0), out=buf30)
        buf33 = reinterpret_tensor(buf7, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf7  # reuse
        # Source Nodes: [attn_weights_10, attn_weights_8, attn_weights_9, full_2, mask_value_1], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg150_1, buf30, buf33, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg150_1
        buf34 = reinterpret_tensor(buf29, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [attn_output_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf27, buf34, 1572864, grid=grid(1572864), stream=stream0)
        buf35 = reinterpret_tensor(buf28, (24, 1024, 64), (65536, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf34, (24, 1024, 64), (65536, 64, 1), 0), out=buf35)
        buf36 = reinterpret_tensor(buf34, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [tensor_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf35, buf36, 1572864, grid=grid(1572864), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (2048, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf36, (2048, 768), (768, 1), 0), arg11_1, out=buf37)
        del arg11_1
        buf38 = reinterpret_tensor(buf37, (2, 1024, 768), (786432, 768, 1), 0); del buf37  # reuse
        buf42 = reinterpret_tensor(buf36, (2, 1024, 768), (786432, 768, 1), 0); del buf36  # reuse
        # Source Nodes: [hidden_states_10, residual_2, residual_3], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf38, arg10_1, buf15, buf22, arg6_1, arg104_1, arg105_1, buf42, 2048, 768, grid=grid(2048), stream=stream0)
        del arg104_1
        del arg105_1
        del arg10_1
        del arg6_1
        buf43 = reinterpret_tensor(buf21, (2048, 3072), (3072, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (2048, 768), (768, 1), 0), arg13_1, out=buf43)
        del arg13_1
        buf44 = reinterpret_tensor(buf43, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf43  # reuse
        # Source Nodes: [add_6, add_7, hidden_states_12, mul_4, mul_5, mul_6, pow_2, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf44, arg12_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg12_1
        buf45 = reinterpret_tensor(buf42, (2048, 768), (768, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (2048, 3072), (3072, 1), 0), arg15_1, out=buf45)
        del arg15_1
        buf49 = reinterpret_tensor(buf22, (2, 1024, 768), (786432, 768, 1), 0); del buf22  # reuse
        # Source Nodes: [hidden_states_16, residual_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf38, buf45, arg14_1, arg106_1, arg107_1, buf49, 2048, 768, grid=grid(2048), stream=stream0)
        del arg106_1
        del arg107_1
        buf50 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg16_1, reinterpret_tensor(buf49, (2048, 768), (768, 1), 0), arg17_1, alpha=1, beta=1, out=buf50)
        del arg16_1
        del arg17_1
        buf51 = reinterpret_tensor(buf49, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf49  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf50, buf51, 1572864, grid=grid(1572864), stream=stream0)
        buf52 = reinterpret_tensor(buf15, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf15  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf50, buf52, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf53 = reinterpret_tensor(buf33, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf33  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf52, (24, 64, 1024), (65536, 1024, 1), 0), out=buf53)
        buf56 = reinterpret_tensor(buf30, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf30  # reuse
        # Source Nodes: [attn_weights_15, attn_weights_16, attn_weights_17, full_4, mask_value_2], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg151_1, buf53, buf56, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg151_1
        buf57 = reinterpret_tensor(buf52, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [attn_output_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf50, buf57, 1572864, grid=grid(1572864), stream=stream0)
        buf58 = reinterpret_tensor(buf51, (24, 1024, 64), (65536, 64, 1), 0); del buf51  # reuse
        # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf56, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf57, (24, 1024, 64), (65536, 64, 1), 0), out=buf58)
        buf59 = reinterpret_tensor(buf57, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf57  # reuse
        # Source Nodes: [tensor_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf58, buf59, 1572864, grid=grid(1572864), stream=stream0)
        buf60 = reinterpret_tensor(buf58, (2048, 768), (768, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (2048, 768), (768, 1), 0), arg19_1, out=buf60)
        del arg19_1
        buf61 = reinterpret_tensor(buf60, (2, 1024, 768), (786432, 768, 1), 0); del buf60  # reuse
        buf65 = reinterpret_tensor(buf59, (2, 1024, 768), (786432, 768, 1), 0); del buf59  # reuse
        # Source Nodes: [hidden_states_18, residual_4, residual_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf61, arg18_1, buf38, buf45, arg14_1, arg108_1, arg109_1, buf65, 2048, 768, grid=grid(2048), stream=stream0)
        del arg108_1
        del arg109_1
        del arg14_1
        del arg18_1
        buf66 = reinterpret_tensor(buf44, (2048, 3072), (3072, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (2048, 768), (768, 1), 0), arg21_1, out=buf66)
        del arg21_1
        buf67 = reinterpret_tensor(buf66, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf66  # reuse
        # Source Nodes: [add_10, add_11, hidden_states_20, mul_10, mul_8, mul_9, pow_3, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf67, arg20_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg20_1
        buf68 = reinterpret_tensor(buf65, (2048, 768), (768, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (2048, 3072), (3072, 1), 0), arg23_1, out=buf68)
        del arg23_1
        buf72 = reinterpret_tensor(buf45, (2, 1024, 768), (786432, 768, 1), 0); del buf45  # reuse
        # Source Nodes: [hidden_states_24, residual_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf61, buf68, arg22_1, arg110_1, arg111_1, buf72, 2048, 768, grid=grid(2048), stream=stream0)
        del arg110_1
        del arg111_1
        buf73 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg24_1, reinterpret_tensor(buf72, (2048, 768), (768, 1), 0), arg25_1, alpha=1, beta=1, out=buf73)
        del arg24_1
        del arg25_1
        buf74 = reinterpret_tensor(buf72, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf72  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf73, buf74, 1572864, grid=grid(1572864), stream=stream0)
        buf75 = reinterpret_tensor(buf38, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf38  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf73, buf75, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf76 = reinterpret_tensor(buf56, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf56  # reuse
        # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf74, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf75, (24, 64, 1024), (65536, 1024, 1), 0), out=buf76)
        buf79 = reinterpret_tensor(buf53, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf53  # reuse
        # Source Nodes: [attn_weights_22, attn_weights_23, attn_weights_24, full_6, mask_value_3], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg152_1, buf76, buf79, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg152_1
        buf80 = reinterpret_tensor(buf75, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf73, buf80, 1572864, grid=grid(1572864), stream=stream0)
        buf81 = reinterpret_tensor(buf74, (24, 1024, 64), (65536, 64, 1), 0); del buf74  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf80, (24, 1024, 64), (65536, 64, 1), 0), out=buf81)
        buf82 = reinterpret_tensor(buf80, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf80  # reuse
        # Source Nodes: [tensor_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf81, buf82, 1572864, grid=grid(1572864), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (2048, 768), (768, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (2048, 768), (768, 1), 0), arg27_1, out=buf83)
        del arg27_1
        buf84 = reinterpret_tensor(buf83, (2, 1024, 768), (786432, 768, 1), 0); del buf83  # reuse
        buf88 = reinterpret_tensor(buf82, (2, 1024, 768), (786432, 768, 1), 0); del buf82  # reuse
        # Source Nodes: [hidden_states_26, residual_6, residual_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf84, arg26_1, buf61, buf68, arg22_1, arg112_1, arg113_1, buf88, 2048, 768, grid=grid(2048), stream=stream0)
        del arg112_1
        del arg113_1
        del arg22_1
        del arg26_1
        buf89 = reinterpret_tensor(buf67, (2048, 3072), (3072, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (2048, 768), (768, 1), 0), arg29_1, out=buf89)
        del arg29_1
        buf90 = reinterpret_tensor(buf89, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf89  # reuse
        # Source Nodes: [add_14, add_15, hidden_states_28, mul_12, mul_13, mul_14, pow_4, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf90, arg28_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg28_1
        buf91 = reinterpret_tensor(buf88, (2048, 768), (768, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (2048, 3072), (3072, 1), 0), arg31_1, out=buf91)
        del arg31_1
        buf95 = reinterpret_tensor(buf68, (2, 1024, 768), (786432, 768, 1), 0); del buf68  # reuse
        # Source Nodes: [hidden_states_32, residual_8], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf84, buf91, arg30_1, arg114_1, arg115_1, buf95, 2048, 768, grid=grid(2048), stream=stream0)
        del arg114_1
        del arg115_1
        buf96 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg32_1, reinterpret_tensor(buf95, (2048, 768), (768, 1), 0), arg33_1, alpha=1, beta=1, out=buf96)
        del arg32_1
        del arg33_1
        buf97 = reinterpret_tensor(buf95, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf95  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf96, buf97, 1572864, grid=grid(1572864), stream=stream0)
        buf98 = reinterpret_tensor(buf61, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf61  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf96, buf98, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf99 = reinterpret_tensor(buf79, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf79  # reuse
        # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf97, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf98, (24, 64, 1024), (65536, 1024, 1), 0), out=buf99)
        buf102 = reinterpret_tensor(buf76, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf76  # reuse
        # Source Nodes: [attn_weights_29, attn_weights_30, attn_weights_31, full_8, mask_value_4], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg153_1, buf99, buf102, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg153_1
        buf103 = reinterpret_tensor(buf98, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf98  # reuse
        # Source Nodes: [attn_output_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf96, buf103, 1572864, grid=grid(1572864), stream=stream0)
        buf104 = reinterpret_tensor(buf97, (24, 1024, 64), (65536, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf102, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf103, (24, 1024, 64), (65536, 64, 1), 0), out=buf104)
        buf105 = reinterpret_tensor(buf103, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [tensor_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf104, buf105, 1572864, grid=grid(1572864), stream=stream0)
        buf106 = reinterpret_tensor(buf104, (2048, 768), (768, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (2048, 768), (768, 1), 0), arg35_1, out=buf106)
        del arg35_1
        buf107 = reinterpret_tensor(buf106, (2, 1024, 768), (786432, 768, 1), 0); del buf106  # reuse
        buf111 = reinterpret_tensor(buf105, (2, 1024, 768), (786432, 768, 1), 0); del buf105  # reuse
        # Source Nodes: [hidden_states_34, residual_8, residual_9], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf107, arg34_1, buf84, buf91, arg30_1, arg116_1, arg117_1, buf111, 2048, 768, grid=grid(2048), stream=stream0)
        del arg116_1
        del arg117_1
        del arg30_1
        del arg34_1
        buf112 = reinterpret_tensor(buf90, (2048, 3072), (3072, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (2048, 768), (768, 1), 0), arg37_1, out=buf112)
        del arg37_1
        buf113 = reinterpret_tensor(buf112, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf112  # reuse
        # Source Nodes: [add_18, add_19, hidden_states_36, mul_16, mul_17, mul_18, pow_5, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf113, arg36_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg36_1
        buf114 = reinterpret_tensor(buf111, (2048, 768), (768, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (2048, 3072), (3072, 1), 0), arg39_1, out=buf114)
        del arg39_1
        buf118 = reinterpret_tensor(buf91, (2, 1024, 768), (786432, 768, 1), 0); del buf91  # reuse
        # Source Nodes: [hidden_states_40, residual_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf107, buf114, arg38_1, arg118_1, arg119_1, buf118, 2048, 768, grid=grid(2048), stream=stream0)
        del arg118_1
        del arg119_1
        buf119 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg40_1, reinterpret_tensor(buf118, (2048, 768), (768, 1), 0), arg41_1, alpha=1, beta=1, out=buf119)
        del arg40_1
        del arg41_1
        buf120 = reinterpret_tensor(buf118, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf118  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf119, buf120, 1572864, grid=grid(1572864), stream=stream0)
        buf121 = reinterpret_tensor(buf84, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf84  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf119, buf121, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf122 = reinterpret_tensor(buf102, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf102  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf121, (24, 64, 1024), (65536, 1024, 1), 0), out=buf122)
        buf125 = reinterpret_tensor(buf99, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf99  # reuse
        # Source Nodes: [attn_weights_36, attn_weights_37, attn_weights_38, full_10, mask_value_5], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg154_1, buf122, buf125, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg154_1
        buf126 = reinterpret_tensor(buf121, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf121  # reuse
        # Source Nodes: [attn_output_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf119, buf126, 1572864, grid=grid(1572864), stream=stream0)
        buf127 = reinterpret_tensor(buf120, (24, 1024, 64), (65536, 64, 1), 0); del buf120  # reuse
        # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf126, (24, 1024, 64), (65536, 64, 1), 0), out=buf127)
        buf128 = reinterpret_tensor(buf126, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf126  # reuse
        # Source Nodes: [tensor_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf127, buf128, 1572864, grid=grid(1572864), stream=stream0)
        buf129 = reinterpret_tensor(buf127, (2048, 768), (768, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (2048, 768), (768, 1), 0), arg43_1, out=buf129)
        del arg43_1
        buf130 = reinterpret_tensor(buf129, (2, 1024, 768), (786432, 768, 1), 0); del buf129  # reuse
        buf134 = reinterpret_tensor(buf128, (2, 1024, 768), (786432, 768, 1), 0); del buf128  # reuse
        # Source Nodes: [hidden_states_42, residual_10, residual_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf130, arg42_1, buf107, buf114, arg38_1, arg120_1, arg121_1, buf134, 2048, 768, grid=grid(2048), stream=stream0)
        del arg120_1
        del arg121_1
        del arg38_1
        del arg42_1
        buf135 = reinterpret_tensor(buf113, (2048, 3072), (3072, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf134, (2048, 768), (768, 1), 0), arg45_1, out=buf135)
        del arg45_1
        buf136 = reinterpret_tensor(buf135, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf135  # reuse
        # Source Nodes: [add_22, add_23, hidden_states_44, mul_20, mul_21, mul_22, pow_6, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf136, arg44_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg44_1
        buf137 = reinterpret_tensor(buf134, (2048, 768), (768, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (2048, 3072), (3072, 1), 0), arg47_1, out=buf137)
        del arg47_1
        buf141 = reinterpret_tensor(buf114, (2, 1024, 768), (786432, 768, 1), 0); del buf114  # reuse
        # Source Nodes: [hidden_states_48, residual_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf130, buf137, arg46_1, arg122_1, arg123_1, buf141, 2048, 768, grid=grid(2048), stream=stream0)
        del arg122_1
        del arg123_1
        buf142 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg48_1, reinterpret_tensor(buf141, (2048, 768), (768, 1), 0), arg49_1, alpha=1, beta=1, out=buf142)
        del arg48_1
        del arg49_1
        buf143 = reinterpret_tensor(buf141, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf141  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf142, buf143, 1572864, grid=grid(1572864), stream=stream0)
        buf144 = reinterpret_tensor(buf107, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf107  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf142, buf144, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf145 = reinterpret_tensor(buf125, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf125  # reuse
        # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf144, (24, 64, 1024), (65536, 1024, 1), 0), out=buf145)
        buf148 = reinterpret_tensor(buf122, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf122  # reuse
        # Source Nodes: [attn_weights_43, attn_weights_44, attn_weights_45, full_12, mask_value_6], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg155_1, buf145, buf148, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg155_1
        buf149 = reinterpret_tensor(buf144, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf144  # reuse
        # Source Nodes: [attn_output_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf142, buf149, 1572864, grid=grid(1572864), stream=stream0)
        buf150 = reinterpret_tensor(buf143, (24, 1024, 64), (65536, 64, 1), 0); del buf143  # reuse
        # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf149, (24, 1024, 64), (65536, 64, 1), 0), out=buf150)
        buf151 = reinterpret_tensor(buf149, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf149  # reuse
        # Source Nodes: [tensor_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf150, buf151, 1572864, grid=grid(1572864), stream=stream0)
        buf152 = reinterpret_tensor(buf150, (2048, 768), (768, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (2048, 768), (768, 1), 0), arg51_1, out=buf152)
        del arg51_1
        buf153 = reinterpret_tensor(buf152, (2, 1024, 768), (786432, 768, 1), 0); del buf152  # reuse
        buf157 = reinterpret_tensor(buf151, (2, 1024, 768), (786432, 768, 1), 0); del buf151  # reuse
        # Source Nodes: [hidden_states_50, residual_12, residual_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf153, arg50_1, buf130, buf137, arg46_1, arg124_1, arg125_1, buf157, 2048, 768, grid=grid(2048), stream=stream0)
        del arg124_1
        del arg125_1
        del arg46_1
        del arg50_1
        buf158 = reinterpret_tensor(buf136, (2048, 3072), (3072, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (2048, 768), (768, 1), 0), arg53_1, out=buf158)
        del arg53_1
        buf159 = reinterpret_tensor(buf158, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf158  # reuse
        # Source Nodes: [add_26, add_27, hidden_states_52, mul_24, mul_25, mul_26, pow_7, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf159, arg52_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg52_1
        buf160 = reinterpret_tensor(buf157, (2048, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (2048, 3072), (3072, 1), 0), arg55_1, out=buf160)
        del arg55_1
        buf164 = reinterpret_tensor(buf137, (2, 1024, 768), (786432, 768, 1), 0); del buf137  # reuse
        # Source Nodes: [hidden_states_56, residual_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf153, buf160, arg54_1, arg126_1, arg127_1, buf164, 2048, 768, grid=grid(2048), stream=stream0)
        del arg126_1
        del arg127_1
        buf165 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg56_1, reinterpret_tensor(buf164, (2048, 768), (768, 1), 0), arg57_1, alpha=1, beta=1, out=buf165)
        del arg56_1
        del arg57_1
        buf166 = reinterpret_tensor(buf164, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf164  # reuse
        # Source Nodes: [attn_weights_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf165, buf166, 1572864, grid=grid(1572864), stream=stream0)
        buf167 = reinterpret_tensor(buf130, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf130  # reuse
        # Source Nodes: [attn_weights_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf165, buf167, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf168 = reinterpret_tensor(buf148, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf148  # reuse
        # Source Nodes: [attn_weights_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf167, (24, 64, 1024), (65536, 1024, 1), 0), out=buf168)
        buf171 = reinterpret_tensor(buf145, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf145  # reuse
        # Source Nodes: [attn_weights_50, attn_weights_51, attn_weights_52, full_14, mask_value_7], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg156_1, buf168, buf171, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg156_1
        buf172 = reinterpret_tensor(buf167, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf167  # reuse
        # Source Nodes: [attn_output_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf165, buf172, 1572864, grid=grid(1572864), stream=stream0)
        buf173 = reinterpret_tensor(buf166, (24, 1024, 64), (65536, 64, 1), 0); del buf166  # reuse
        # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf172, (24, 1024, 64), (65536, 64, 1), 0), out=buf173)
        buf174 = reinterpret_tensor(buf172, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf172  # reuse
        # Source Nodes: [tensor_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf173, buf174, 1572864, grid=grid(1572864), stream=stream0)
        buf175 = reinterpret_tensor(buf173, (2048, 768), (768, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (2048, 768), (768, 1), 0), arg59_1, out=buf175)
        del arg59_1
        buf176 = reinterpret_tensor(buf175, (2, 1024, 768), (786432, 768, 1), 0); del buf175  # reuse
        buf180 = reinterpret_tensor(buf174, (2, 1024, 768), (786432, 768, 1), 0); del buf174  # reuse
        # Source Nodes: [hidden_states_58, residual_14, residual_15], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf176, arg58_1, buf153, buf160, arg54_1, arg128_1, arg129_1, buf180, 2048, 768, grid=grid(2048), stream=stream0)
        del arg128_1
        del arg129_1
        del arg54_1
        del arg58_1
        buf181 = reinterpret_tensor(buf159, (2048, 3072), (3072, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (2048, 768), (768, 1), 0), arg61_1, out=buf181)
        del arg61_1
        buf182 = reinterpret_tensor(buf181, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf181  # reuse
        # Source Nodes: [add_30, add_31, hidden_states_60, mul_28, mul_29, mul_30, pow_8, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf182, arg60_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg60_1
        buf183 = reinterpret_tensor(buf180, (2048, 768), (768, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (2048, 3072), (3072, 1), 0), arg63_1, out=buf183)
        del arg63_1
        buf187 = reinterpret_tensor(buf160, (2, 1024, 768), (786432, 768, 1), 0); del buf160  # reuse
        # Source Nodes: [hidden_states_64, residual_16], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf176, buf183, arg62_1, arg130_1, arg131_1, buf187, 2048, 768, grid=grid(2048), stream=stream0)
        del arg130_1
        del arg131_1
        buf188 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg64_1, reinterpret_tensor(buf187, (2048, 768), (768, 1), 0), arg65_1, alpha=1, beta=1, out=buf188)
        del arg64_1
        del arg65_1
        buf189 = reinterpret_tensor(buf187, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf187  # reuse
        # Source Nodes: [attn_weights_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf188, buf189, 1572864, grid=grid(1572864), stream=stream0)
        buf190 = reinterpret_tensor(buf153, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf153  # reuse
        # Source Nodes: [attn_weights_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf188, buf190, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf191 = reinterpret_tensor(buf171, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf171  # reuse
        # Source Nodes: [attn_weights_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf190, (24, 64, 1024), (65536, 1024, 1), 0), out=buf191)
        buf194 = reinterpret_tensor(buf168, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf168  # reuse
        # Source Nodes: [attn_weights_57, attn_weights_58, attn_weights_59, full_16, mask_value_8], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg157_1, buf191, buf194, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg157_1
        buf195 = reinterpret_tensor(buf190, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf188, buf195, 1572864, grid=grid(1572864), stream=stream0)
        buf196 = reinterpret_tensor(buf189, (24, 1024, 64), (65536, 64, 1), 0); del buf189  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf195, (24, 1024, 64), (65536, 64, 1), 0), out=buf196)
        buf197 = reinterpret_tensor(buf195, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf195  # reuse
        # Source Nodes: [tensor_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf196, buf197, 1572864, grid=grid(1572864), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (2048, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (2048, 768), (768, 1), 0), arg67_1, out=buf198)
        del arg67_1
        buf199 = reinterpret_tensor(buf198, (2, 1024, 768), (786432, 768, 1), 0); del buf198  # reuse
        buf203 = reinterpret_tensor(buf197, (2, 1024, 768), (786432, 768, 1), 0); del buf197  # reuse
        # Source Nodes: [hidden_states_66, residual_16, residual_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf199, arg66_1, buf176, buf183, arg62_1, arg132_1, arg133_1, buf203, 2048, 768, grid=grid(2048), stream=stream0)
        del arg132_1
        del arg133_1
        del arg62_1
        del arg66_1
        buf204 = reinterpret_tensor(buf182, (2048, 3072), (3072, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (2048, 768), (768, 1), 0), arg69_1, out=buf204)
        del arg69_1
        buf205 = reinterpret_tensor(buf204, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf204  # reuse
        # Source Nodes: [add_34, add_35, hidden_states_68, mul_32, mul_33, mul_34, pow_9, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf205, arg68_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg68_1
        buf206 = reinterpret_tensor(buf203, (2048, 768), (768, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (2048, 3072), (3072, 1), 0), arg71_1, out=buf206)
        del arg71_1
        buf210 = reinterpret_tensor(buf183, (2, 1024, 768), (786432, 768, 1), 0); del buf183  # reuse
        # Source Nodes: [hidden_states_72, residual_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf199, buf206, arg70_1, arg134_1, arg135_1, buf210, 2048, 768, grid=grid(2048), stream=stream0)
        del arg134_1
        del arg135_1
        buf211 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg72_1, reinterpret_tensor(buf210, (2048, 768), (768, 1), 0), arg73_1, alpha=1, beta=1, out=buf211)
        del arg72_1
        del arg73_1
        buf212 = reinterpret_tensor(buf210, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf210  # reuse
        # Source Nodes: [attn_weights_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf211, buf212, 1572864, grid=grid(1572864), stream=stream0)
        buf213 = reinterpret_tensor(buf176, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf176  # reuse
        # Source Nodes: [attn_weights_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf211, buf213, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf214 = reinterpret_tensor(buf194, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf194  # reuse
        # Source Nodes: [attn_weights_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf212, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf213, (24, 64, 1024), (65536, 1024, 1), 0), out=buf214)
        buf217 = reinterpret_tensor(buf191, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf191  # reuse
        # Source Nodes: [attn_weights_64, attn_weights_65, attn_weights_66, full_18, mask_value_9], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg158_1, buf214, buf217, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg158_1
        buf218 = reinterpret_tensor(buf213, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [attn_output_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf211, buf218, 1572864, grid=grid(1572864), stream=stream0)
        buf219 = reinterpret_tensor(buf212, (24, 1024, 64), (65536, 64, 1), 0); del buf212  # reuse
        # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf217, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf218, (24, 1024, 64), (65536, 64, 1), 0), out=buf219)
        buf220 = reinterpret_tensor(buf218, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf218  # reuse
        # Source Nodes: [tensor_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf219, buf220, 1572864, grid=grid(1572864), stream=stream0)
        buf221 = reinterpret_tensor(buf219, (2048, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (2048, 768), (768, 1), 0), arg75_1, out=buf221)
        del arg75_1
        buf222 = reinterpret_tensor(buf221, (2, 1024, 768), (786432, 768, 1), 0); del buf221  # reuse
        buf226 = reinterpret_tensor(buf220, (2, 1024, 768), (786432, 768, 1), 0); del buf220  # reuse
        # Source Nodes: [hidden_states_74, residual_18, residual_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf222, arg74_1, buf199, buf206, arg70_1, arg136_1, arg137_1, buf226, 2048, 768, grid=grid(2048), stream=stream0)
        del arg136_1
        del arg137_1
        del arg70_1
        del arg74_1
        buf227 = reinterpret_tensor(buf205, (2048, 3072), (3072, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (2048, 768), (768, 1), 0), arg77_1, out=buf227)
        del arg77_1
        buf228 = reinterpret_tensor(buf227, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf227  # reuse
        # Source Nodes: [add_38, add_39, hidden_states_76, mul_36, mul_37, mul_38, pow_10, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf228, arg76_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg76_1
        buf229 = reinterpret_tensor(buf226, (2048, 768), (768, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (2048, 3072), (3072, 1), 0), arg79_1, out=buf229)
        del arg79_1
        buf233 = reinterpret_tensor(buf206, (2, 1024, 768), (786432, 768, 1), 0); del buf206  # reuse
        # Source Nodes: [hidden_states_80, residual_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf222, buf229, arg78_1, arg138_1, arg139_1, buf233, 2048, 768, grid=grid(2048), stream=stream0)
        del arg138_1
        del arg139_1
        buf234 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg80_1, reinterpret_tensor(buf233, (2048, 768), (768, 1), 0), arg81_1, alpha=1, beta=1, out=buf234)
        del arg80_1
        del arg81_1
        buf235 = reinterpret_tensor(buf233, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf233  # reuse
        # Source Nodes: [attn_weights_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf234, buf235, 1572864, grid=grid(1572864), stream=stream0)
        buf236 = reinterpret_tensor(buf199, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf199  # reuse
        # Source Nodes: [attn_weights_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf234, buf236, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf237 = reinterpret_tensor(buf217, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf217  # reuse
        # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf236, (24, 64, 1024), (65536, 1024, 1), 0), out=buf237)
        buf240 = reinterpret_tensor(buf214, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf214  # reuse
        # Source Nodes: [attn_weights_71, attn_weights_72, attn_weights_73, full_20, mask_value_10], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg159_1, buf237, buf240, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg159_1
        buf241 = reinterpret_tensor(buf236, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [attn_output_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf234, buf241, 1572864, grid=grid(1572864), stream=stream0)
        buf242 = reinterpret_tensor(buf235, (24, 1024, 64), (65536, 64, 1), 0); del buf235  # reuse
        # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf240, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf241, (24, 1024, 64), (65536, 64, 1), 0), out=buf242)
        buf243 = reinterpret_tensor(buf241, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf241  # reuse
        # Source Nodes: [tensor_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf242, buf243, 1572864, grid=grid(1572864), stream=stream0)
        buf244 = reinterpret_tensor(buf242, (2048, 768), (768, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (2048, 768), (768, 1), 0), arg83_1, out=buf244)
        del arg83_1
        buf245 = reinterpret_tensor(buf244, (2, 1024, 768), (786432, 768, 1), 0); del buf244  # reuse
        buf249 = reinterpret_tensor(buf243, (2, 1024, 768), (786432, 768, 1), 0); del buf243  # reuse
        # Source Nodes: [hidden_states_82, residual_20, residual_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf245, arg82_1, buf222, buf229, arg78_1, arg140_1, arg141_1, buf249, 2048, 768, grid=grid(2048), stream=stream0)
        del arg140_1
        del arg141_1
        del arg78_1
        del arg82_1
        buf250 = reinterpret_tensor(buf228, (2048, 3072), (3072, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (2048, 768), (768, 1), 0), arg85_1, out=buf250)
        del arg85_1
        buf251 = reinterpret_tensor(buf250, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf250  # reuse
        # Source Nodes: [add_42, add_43, hidden_states_84, mul_40, mul_41, mul_42, pow_11, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf251, arg84_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg84_1
        buf252 = reinterpret_tensor(buf249, (2048, 768), (768, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (2048, 3072), (3072, 1), 0), arg87_1, out=buf252)
        del arg87_1
        buf256 = reinterpret_tensor(buf229, (2, 1024, 768), (786432, 768, 1), 0); del buf229  # reuse
        # Source Nodes: [hidden_states_88, residual_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf245, buf252, arg86_1, arg142_1, arg143_1, buf256, 2048, 768, grid=grid(2048), stream=stream0)
        del arg142_1
        del arg143_1
        buf257 = empty((2048, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg88_1, reinterpret_tensor(buf256, (2048, 768), (768, 1), 0), arg89_1, alpha=1, beta=1, out=buf257)
        del arg88_1
        del arg89_1
        buf258 = reinterpret_tensor(buf256, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf256  # reuse
        # Source Nodes: [attn_weights_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf257, buf258, 1572864, grid=grid(1572864), stream=stream0)
        buf259 = reinterpret_tensor(buf222, (2, 12, 64, 1024), (786432, 65536, 1024, 1), 0); del buf222  # reuse
        # Source Nodes: [attn_weights_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf257, buf259, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        buf260 = reinterpret_tensor(buf240, (24, 1024, 1024), (1048576, 1024, 1), 0); del buf240  # reuse
        # Source Nodes: [attn_weights_77], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf258, (24, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf259, (24, 64, 1024), (65536, 1024, 1), 0), out=buf260)
        buf263 = reinterpret_tensor(buf237, (2, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf237  # reuse
        # Source Nodes: [attn_weights_78, attn_weights_79, attn_weights_80, full_22, mask_value_11], Original ATen: [aten._softmax, aten._to_copy, aten.div, aten.full, aten.where]
        triton_per_fused__softmax__to_copy_div_full_where_3.run(arg160_1, buf260, buf263, 24576, 1024, grid=grid(24576), stream=stream0)
        del arg160_1
        del buf260
        buf264 = reinterpret_tensor(buf259, (2, 12, 1024, 64), (786432, 65536, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [attn_output_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf257, buf264, 1572864, grid=grid(1572864), stream=stream0)
        buf265 = reinterpret_tensor(buf258, (24, 1024, 64), (65536, 64, 1), 0); del buf258  # reuse
        # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf263, (24, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf264, (24, 1024, 64), (65536, 64, 1), 0), out=buf265)
        del buf263
        buf266 = reinterpret_tensor(buf264, (2, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf264  # reuse
        # Source Nodes: [tensor_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf265, buf266, 1572864, grid=grid(1572864), stream=stream0)
        buf267 = reinterpret_tensor(buf265, (2048, 768), (768, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (2048, 768), (768, 1), 0), arg91_1, out=buf267)
        del arg91_1
        buf268 = reinterpret_tensor(buf267, (2, 1024, 768), (786432, 768, 1), 0); del buf267  # reuse
        buf272 = reinterpret_tensor(buf266, (2, 1024, 768), (786432, 768, 1), 0); del buf266  # reuse
        # Source Nodes: [hidden_states_90, residual_22, residual_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf268, arg90_1, buf245, buf252, arg86_1, arg144_1, arg145_1, buf272, 2048, 768, grid=grid(2048), stream=stream0)
        del arg144_1
        del arg145_1
        del arg86_1
        del arg90_1
        del buf245
        buf273 = reinterpret_tensor(buf251, (2048, 3072), (3072, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (2048, 768), (768, 1), 0), arg93_1, out=buf273)
        del arg93_1
        buf274 = reinterpret_tensor(buf273, (2, 1024, 3072), (3145728, 3072, 1), 0); del buf273  # reuse
        # Source Nodes: [add_46, add_47, hidden_states_92, mul_44, mul_45, mul_46, pow_12, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_7.run(buf274, arg92_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg92_1
        buf275 = reinterpret_tensor(buf272, (2048, 768), (768, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (2048, 3072), (3072, 1), 0), arg95_1, out=buf275)
        del arg95_1
        del buf274
        buf279 = reinterpret_tensor(buf252, (2, 1024, 768), (786432, 768, 1), 0); del buf252  # reuse
        # Source Nodes: [hidden_states_95, l__mod___transformer_ln_f], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf268, buf275, arg94_1, arg146_1, arg147_1, buf279, 2048, 768, grid=grid(2048), stream=stream0)
        del arg146_1
        del arg147_1
        del arg94_1
        del buf268
        del buf275
        buf280 = empty_strided((768, 50260), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_10.run(arg148_1, buf280, 38599680, grid=grid(38599680), stream=stream0)
        del arg148_1
        buf281 = empty((2048, 50260), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (2048, 768), (768, 1), 0), buf280, out=buf281)
        return (reinterpret_tensor(buf281, (2, 1024, 50257), (51466240, 50260, 1), 0), reinterpret_tensor(buf4, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf4, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf27, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf27, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf50, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf50, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf73, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf73, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf96, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf96, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf119, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf119, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf142, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf142, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf165, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf165, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf188, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf188, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf211, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf211, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf234, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf234, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), reinterpret_tensor(buf257, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 768), reinterpret_tensor(buf257, (2, 12, 1024, 64), (2359296, 64, 2304, 1), 1536), )


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
    arg48_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg150_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg151_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg152_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg153_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg154_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg155_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg156_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg157_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg158_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg159_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg160_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg161_1 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_GPT2', benchmark_compiled_module)
