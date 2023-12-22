
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


# kernel path: /tmp/torchinductor_youkaichao/35/c35q7zqr3zxl2pjfztof3n55cvwhc2idq7exjo2esqa5b5pokahf.py
# Source Nodes: [add_2, hidden_states_1, inputs_embeds, pos_embeds, positions, residual], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.sub]
# add_2 => add_2
# hidden_states_1 => add_4, add_5, mul_1, mul_2, rsqrt, sub_2, var_mean
# inputs_embeds => embedding
# pos_embeds => embedding_1
# positions => sub_1
# residual => add_3
triton_red_fused_add_embedding_native_layer_norm_sub_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_native_layer_norm_sub_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (1536 + r1 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50272
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 50272), "index out of bounds: 0 <= tmp3 < 50272")
        tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
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
        r1 = rindex
        tmp15 = tl.load(in_ptr2 + (1536 + r1 + (768*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 + 50272
        tmp12 = tmp0 < 0
        tmp13 = tl.where(tmp12, tmp11, tmp0)
        tl.device_assert((0 <= tmp13) & (tmp13 < 50272), "index out of bounds: 0 <= tmp13 < 50272")
        tmp14 = tl.load(in_ptr1 + (r1 + (768*tmp13)), rmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/cl/cclhfup4hrjpg2kbyblw7gcqvxsddm4sulwlxdb5c7cmjv75zsru.py
# Source Nodes: [key_states], Original ATen: [aten.clone]
# key_states => clone
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
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clm7fd3gitcd5hgwynkhj7fq5dhdp7qisq46tdfxbhsy3a53gdpv.py
# Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# contiguous_2 => clone_2
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
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxdeq46eydwhag4bwtjfd367lqutox5dygctchzeu6j5cluiflh.py
# Source Nodes: [attn_weights_4], Original ATen: [aten._softmax]
# attn_weights_4 => amax, div, exp, sub_3, sum_1
triton_red_fused__softmax_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    _tmp10 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r2
        tmp2 = 1 + x0
        tmp3 = tmp1 < tmp2
        tmp4 = 0.0
        tmp5 = -3.4028234663852886e+38
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tmp0 + tmp6
        tmp8 = triton_helpers.maximum(tmp7, tmp5)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = triton_helpers.maximum(_tmp10, tmp9)
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = triton_helpers.max2(_tmp10, 1)[:, None]
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = r2
        tmp14 = 1 + x0
        tmp15 = tmp13 < tmp14
        tmp16 = 0.0
        tmp17 = -3.4028234663852886e+38
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tmp12 + tmp18
        tmp20 = triton_helpers.maximum(tmp19, tmp17)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp26 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp27 = r2
        tmp28 = 1 + x0
        tmp29 = tmp27 < tmp28
        tmp30 = 0.0
        tmp31 = -3.4028234663852886e+38
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp33 = tmp26 + tmp32
        tmp34 = triton_helpers.maximum(tmp33, tmp31)
        tmp35 = tmp34 - tmp10
        tmp36 = tl.exp(tmp35)
        tmp37 = tmp36 / tmp24
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmogugeol3tlehpfwakizwmxdcuybwl27puyz4ym7ahyv4ub7em.py
# Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# attn_output_3 => clone_4
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
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (131072*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmsgcngs3h6k7y2vsqisnxercayugy6lk7pe3lqwdyqerw45w6kg.py
# Source Nodes: [add_2, hidden_states_4, hidden_states_6, inputs_embeds, pos_embeds, positions, residual], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.sub]
# add_2 => add_2
# hidden_states_4 => add_7
# hidden_states_6 => add_8, add_9, mul_4, mul_5, rsqrt_1, sub_4, var_mean_1
# inputs_embeds => embedding
# pos_embeds => embedding_1
# positions => sub_1
# residual => add_3
triton_per_fused_add_embedding_native_layer_norm_sub_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_sub_5', 'mutated_arg_names': ['in_out_ptr0']}
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
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (1536 + r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 50272
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 50272), "index out of bounds: 0 <= tmp3 < 50272")
    tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
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
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp10, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55uapboz3i6pmsr6ixn2koe7tybbunkmaazj66ner22nj4e6qew.py
# Source Nodes: [hidden_states_8], Original ATen: [aten.relu]
# hidden_states_8 => relu
triton_poi_fused_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7cercy22rdyhewxsja5dwhnvcp66nxxrfvomfcvhld526ycsis.py
# Source Nodes: [hidden_states_13], Original ATen: [aten.native_layer_norm]
# hidden_states_13 => add_11, add_12, mul_6, mul_7, rsqrt_2, sub_5, var_mean_2
triton_per_fused_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xo/cxobkrkgclvdc3uwfeaekrrfbvql63grkmbg6dt6ftuqazsj4lt5.py
# Source Nodes: [hidden_states_16, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
# hidden_states_16 => add_14
# hidden_states_18 => add_15, add_16, mul_10, mul_9, rsqrt_3, sub_7, var_mean_3
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
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
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/av/cav5ueikzoz2bmycc4mpizve4zadb7h3dsiejyoo6drj4whzmxvv.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_12, exp_12, sub_39, sum_13
triton_red_fused__log_softmax_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2047
    rnumel = 50272
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
        tmp0 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvx2prrmjrrukf6l5zqnhgnban75y57tvqvwi5qrxu2nuoxj263d.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_2, div_12, full_default_17, ne_1, ne_2, neg, sum_14, sum_15, where_3
triton_red_fused_nll_loss_forward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2047
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = tmp4 + 50272
        tmp6 = tmp4 < 0
        tmp7 = tl.where(tmp6, tmp5, tmp4)
        tl.device_assert((0 <= tmp7) & (tmp7 < 50272), "index out of bounds: 0 <= tmp7 < 50272")
        tmp8 = tl.load(in_ptr1 + (tmp7 + (50272*r0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp12 = tl.log(tmp11)
        tmp13 = tmp10 - tmp12
        tmp14 = -tmp13
        tmp15 = 0.0
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
        tmp20 = tmp2.to(tl.int64)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp24 = tmp22.to(tl.float32)
    tmp25 = tmp18 / tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp25, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2050, 768), (768, 1))
    assert_size_stride(arg1_1, (50272, 768), (768, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, 768), (768, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, 768), (768, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (3072, 768), (768, 1))
    assert_size_stride(arg15_1, (3072, ), (1, ))
    assert_size_stride(arg16_1, (768, 3072), (3072, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, 768), (768, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, 768), (768, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (3072, 768), (768, 1))
    assert_size_stride(arg31_1, (3072, ), (1, ))
    assert_size_stride(arg32_1, (768, 3072), (3072, 1))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, 768), (768, 1))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, 768), (768, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, 768), (768, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (3072, 768), (768, 1))
    assert_size_stride(arg47_1, (3072, ), (1, ))
    assert_size_stride(arg48_1, (768, 3072), (3072, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, 768), (768, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (3072, 768), (768, 1))
    assert_size_stride(arg63_1, (3072, ), (1, ))
    assert_size_stride(arg64_1, (768, 3072), (3072, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, 768), (768, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (3072, 768), (768, 1))
    assert_size_stride(arg79_1, (3072, ), (1, ))
    assert_size_stride(arg80_1, (768, 3072), (3072, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, 768), (768, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (3072, 768), (768, 1))
    assert_size_stride(arg95_1, (3072, ), (1, ))
    assert_size_stride(arg96_1, (768, 3072), (3072, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 768), (768, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (3072, 768), (768, 1))
    assert_size_stride(arg111_1, (3072, ), (1, ))
    assert_size_stride(arg112_1, (768, 3072), (3072, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 768), (768, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, 768), (768, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (3072, 768), (768, 1))
    assert_size_stride(arg127_1, (3072, ), (1, ))
    assert_size_stride(arg128_1, (768, 3072), (3072, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, 768), (768, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (3072, 768), (768, 1))
    assert_size_stride(arg143_1, (3072, ), (1, ))
    assert_size_stride(arg144_1, (768, 3072), (3072, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, 768), (768, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, 768), (768, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, 768), (768, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, 768), (768, 1))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (3072, 768), (768, 1))
    assert_size_stride(arg159_1, (3072, ), (1, ))
    assert_size_stride(arg160_1, (768, 3072), (3072, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, 768), (768, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, 768), (768, 1))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, 768), (768, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (3072, 768), (768, 1))
    assert_size_stride(arg175_1, (3072, ), (1, ))
    assert_size_stride(arg176_1, (768, 3072), (3072, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, 768), (768, 1))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (3072, 768), (768, 1))
    assert_size_stride(arg191_1, (3072, ), (1, ))
    assert_size_stride(arg192_1, (768, 3072), (3072, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (50272, 768), (768, 1))
    assert_size_stride(arg197_1, (1, 2048), (2048, 1))
    assert_size_stride(arg198_1, (1, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((1, 2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_2, hidden_states_1, inputs_embeds, pos_embeds, positions, residual], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.sub]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_native_layer_norm_sub_0.run(arg197_1, arg1_1, arg0_1, arg2_1, arg3_1, buf3, 2048, 768, grid=grid(2048), stream=stream0)
        del arg2_1
        del arg3_1
        buf4 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 768), (768, 1), 0), reinterpret_tensor(arg4_1, (768, 768), (1, 768), 0), out=buf4)
        del arg4_1
        buf5 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), out=buf5)
        del arg6_1
        buf6 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf5, arg7_1, buf6, 1572864, grid=grid(1572864), stream=stream0)
        del arg7_1
        buf7 = reinterpret_tensor(buf5, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf4, arg5_1, buf7, 1572864, grid=grid(1572864), stream=stream0)
        del arg5_1
        buf8 = empty((12, 2048, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf6, (12, 64, 2048), (131072, 1, 64), 0), out=buf8)
        buf13 = empty((12, 2048, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf8, buf13, 24576, 2048, grid=grid(24576), stream=stream0)
        buf11 = reinterpret_tensor(buf7, (2048, 768), (768, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), out=buf11)
        del arg8_1
        buf12 = reinterpret_tensor(buf3, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf3  # reuse
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf11, arg9_1, buf12, 1572864, grid=grid(1572864), stream=stream0)
        del arg9_1
        buf14 = reinterpret_tensor(buf11, (12, 2048, 64), (131072, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [attn_output, attn_weights_4], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf13, reinterpret_tensor(buf12, (12, 2048, 64), (131072, 64, 1), 0), out=buf14)
        buf15 = reinterpret_tensor(buf4, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf14, buf15, 1572864, grid=grid(1572864), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (2048, 768), (768, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (2048, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), out=buf16)
        del arg10_1
        buf17 = reinterpret_tensor(buf16, (1, 2048, 768), (1572864, 768, 1), 0); del buf16  # reuse
        buf21 = reinterpret_tensor(buf15, (2048, 768), (768, 1), 0); del buf15  # reuse
        # Source Nodes: [add_2, hidden_states_4, hidden_states_6, inputs_embeds, pos_embeds, positions, residual], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm, aten.sub]
        triton_per_fused_add_embedding_native_layer_norm_sub_5.run(buf17, arg197_1, arg1_1, arg0_1, arg11_1, arg12_1, arg13_1, buf21, 2048, 768, grid=grid(2048), stream=stream0)
        del arg0_1
        del arg11_1
        del arg12_1
        del arg13_1
        del arg197_1
        del arg1_1
        buf22 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_6], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf21, reinterpret_tensor(arg14_1, (768, 3072), (1, 768), 0), out=buf22)
        del arg14_1
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf23, arg15_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg15_1
        buf24 = buf21; del buf21  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.relu]
        extern_kernels.mm(buf23, reinterpret_tensor(arg16_1, (3072, 768), (1, 3072), 0), out=buf24)
        del arg16_1
        buf28 = empty((1, 2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_13], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf17, buf24, arg17_1, arg18_1, arg19_1, buf28, 2048, 768, grid=grid(2048), stream=stream0)
        del arg18_1
        del arg19_1
        buf29 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (2048, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 768), (1, 768), 0), out=buf29)
        del arg20_1
        buf30 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (2048, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), out=buf30)
        del arg22_1
        buf31 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf30, arg23_1, buf31, 1572864, grid=grid(1572864), stream=stream0)
        del arg23_1
        buf32 = reinterpret_tensor(buf30, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf29, arg21_1, buf32, 1572864, grid=grid(1572864), stream=stream0)
        del arg21_1
        buf33 = buf13; del buf13  # reuse
        # Source Nodes: [attn_weights_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf31, (12, 64, 2048), (131072, 1, 64), 0), out=buf33)
        buf38 = buf8; del buf8  # reuse
        # Source Nodes: [attn_weights_9], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf33, buf38, 24576, 2048, grid=grid(24576), stream=stream0)
        buf36 = reinterpret_tensor(buf32, (2048, 768), (768, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (2048, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), out=buf36)
        del arg24_1
        buf37 = reinterpret_tensor(buf28, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf36, arg25_1, buf37, 1572864, grid=grid(1572864), stream=stream0)
        del arg25_1
        buf39 = reinterpret_tensor(buf36, (12, 2048, 64), (131072, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [attn_output_5, attn_weights_9], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf38, reinterpret_tensor(buf37, (12, 2048, 64), (131072, 64, 1), 0), out=buf39)
        buf40 = reinterpret_tensor(buf29, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf39, buf40, 1572864, grid=grid(1572864), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (2048, 768), (768, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (2048, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), out=buf41)
        del arg26_1
        buf42 = reinterpret_tensor(buf41, (1, 2048, 768), (1572864, 768, 1), 0); del buf41  # reuse
        buf46 = reinterpret_tensor(buf40, (2048, 768), (768, 1), 0); del buf40  # reuse
        # Source Nodes: [hidden_states_16, hidden_states_18], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf42, buf17, buf24, arg17_1, arg27_1, arg28_1, arg29_1, buf46, 2048, 768, grid=grid(2048), stream=stream0)
        del arg17_1
        del arg27_1
        del arg28_1
        del arg29_1
        buf47 = buf23; del buf23  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf46, reinterpret_tensor(arg30_1, (768, 3072), (1, 768), 0), out=buf47)
        del arg30_1
        buf48 = buf47; del buf47  # reuse
        # Source Nodes: [hidden_states_20], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf48, arg31_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg31_1
        buf49 = buf46; del buf46  # reuse
        # Source Nodes: [hidden_states_20], Original ATen: [aten.relu]
        extern_kernels.mm(buf48, reinterpret_tensor(arg32_1, (3072, 768), (1, 3072), 0), out=buf49)
        del arg32_1
        buf53 = reinterpret_tensor(buf24, (1, 2048, 768), (1572864, 768, 1), 0); del buf24  # reuse
        # Source Nodes: [hidden_states_25], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf42, buf49, arg33_1, arg34_1, arg35_1, buf53, 2048, 768, grid=grid(2048), stream=stream0)
        del arg34_1
        del arg35_1
        buf54 = reinterpret_tensor(buf17, (2048, 768), (768, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (2048, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 768), (1, 768), 0), out=buf54)
        del arg36_1
        buf55 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (2048, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), out=buf55)
        del arg38_1
        buf56 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf55, arg39_1, buf56, 1572864, grid=grid(1572864), stream=stream0)
        del arg39_1
        buf57 = reinterpret_tensor(buf55, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf54, arg37_1, buf57, 1572864, grid=grid(1572864), stream=stream0)
        del arg37_1
        buf58 = buf38; del buf38  # reuse
        # Source Nodes: [attn_weights_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf56, (12, 64, 2048), (131072, 1, 64), 0), out=buf58)
        buf63 = buf33; del buf33  # reuse
        # Source Nodes: [attn_weights_14], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf58, buf63, 24576, 2048, grid=grid(24576), stream=stream0)
        buf61 = reinterpret_tensor(buf57, (2048, 768), (768, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (2048, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), out=buf61)
        del arg40_1
        buf62 = reinterpret_tensor(buf53, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf53  # reuse
        # Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf61, arg41_1, buf62, 1572864, grid=grid(1572864), stream=stream0)
        del arg41_1
        buf64 = reinterpret_tensor(buf61, (12, 2048, 64), (131072, 64, 1), 0); del buf61  # reuse
        # Source Nodes: [attn_output_10, attn_weights_14], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf63, reinterpret_tensor(buf62, (12, 2048, 64), (131072, 64, 1), 0), out=buf64)
        buf65 = reinterpret_tensor(buf54, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf54  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf64, buf65, 1572864, grid=grid(1572864), stream=stream0)
        buf66 = reinterpret_tensor(buf64, (2048, 768), (768, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (2048, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), out=buf66)
        del arg42_1
        buf67 = reinterpret_tensor(buf66, (1, 2048, 768), (1572864, 768, 1), 0); del buf66  # reuse
        buf71 = reinterpret_tensor(buf65, (2048, 768), (768, 1), 0); del buf65  # reuse
        # Source Nodes: [hidden_states_28, hidden_states_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf67, buf42, buf49, arg33_1, arg43_1, arg44_1, arg45_1, buf71, 2048, 768, grid=grid(2048), stream=stream0)
        del arg33_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf72 = buf48; del buf48  # reuse
        # Source Nodes: [hidden_states_30], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf71, reinterpret_tensor(arg46_1, (768, 3072), (1, 768), 0), out=buf72)
        del arg46_1
        buf73 = buf72; del buf72  # reuse
        # Source Nodes: [hidden_states_32], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf73, arg47_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg47_1
        buf74 = buf71; del buf71  # reuse
        # Source Nodes: [hidden_states_32], Original ATen: [aten.relu]
        extern_kernels.mm(buf73, reinterpret_tensor(arg48_1, (3072, 768), (1, 3072), 0), out=buf74)
        del arg48_1
        buf78 = reinterpret_tensor(buf49, (1, 2048, 768), (1572864, 768, 1), 0); del buf49  # reuse
        # Source Nodes: [hidden_states_37], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf67, buf74, arg49_1, arg50_1, arg51_1, buf78, 2048, 768, grid=grid(2048), stream=stream0)
        del arg50_1
        del arg51_1
        buf79 = reinterpret_tensor(buf42, (2048, 768), (768, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (2048, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 768), (1, 768), 0), out=buf79)
        del arg52_1
        buf80 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (2048, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), out=buf80)
        del arg54_1
        buf81 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf80, arg55_1, buf81, 1572864, grid=grid(1572864), stream=stream0)
        del arg55_1
        buf82 = reinterpret_tensor(buf80, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf80  # reuse
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf79, arg53_1, buf82, 1572864, grid=grid(1572864), stream=stream0)
        del arg53_1
        buf83 = buf63; del buf63  # reuse
        # Source Nodes: [attn_weights_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf81, (12, 64, 2048), (131072, 1, 64), 0), out=buf83)
        buf88 = buf58; del buf58  # reuse
        # Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf83, buf88, 24576, 2048, grid=grid(24576), stream=stream0)
        buf86 = reinterpret_tensor(buf82, (2048, 768), (768, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (2048, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), out=buf86)
        del arg56_1
        buf87 = reinterpret_tensor(buf78, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf86, arg57_1, buf87, 1572864, grid=grid(1572864), stream=stream0)
        del arg57_1
        buf89 = reinterpret_tensor(buf86, (12, 2048, 64), (131072, 64, 1), 0); del buf86  # reuse
        # Source Nodes: [attn_output_15, attn_weights_19], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf88, reinterpret_tensor(buf87, (12, 2048, 64), (131072, 64, 1), 0), out=buf89)
        buf90 = reinterpret_tensor(buf79, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf79  # reuse
        # Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf89, buf90, 1572864, grid=grid(1572864), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (2048, 768), (768, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (2048, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), out=buf91)
        del arg58_1
        buf92 = reinterpret_tensor(buf91, (1, 2048, 768), (1572864, 768, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf90, (2048, 768), (768, 1), 0); del buf90  # reuse
        # Source Nodes: [hidden_states_40, hidden_states_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf92, buf67, buf74, arg49_1, arg59_1, arg60_1, arg61_1, buf96, 2048, 768, grid=grid(2048), stream=stream0)
        del arg49_1
        del arg59_1
        del arg60_1
        del arg61_1
        buf97 = buf73; del buf73  # reuse
        # Source Nodes: [hidden_states_42], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf96, reinterpret_tensor(arg62_1, (768, 3072), (1, 768), 0), out=buf97)
        del arg62_1
        buf98 = buf97; del buf97  # reuse
        # Source Nodes: [hidden_states_44], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf98, arg63_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg63_1
        buf99 = buf96; del buf96  # reuse
        # Source Nodes: [hidden_states_44], Original ATen: [aten.relu]
        extern_kernels.mm(buf98, reinterpret_tensor(arg64_1, (3072, 768), (1, 3072), 0), out=buf99)
        del arg64_1
        buf103 = reinterpret_tensor(buf74, (1, 2048, 768), (1572864, 768, 1), 0); del buf74  # reuse
        # Source Nodes: [hidden_states_49], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf92, buf99, arg65_1, arg66_1, arg67_1, buf103, 2048, 768, grid=grid(2048), stream=stream0)
        del arg66_1
        del arg67_1
        buf104 = reinterpret_tensor(buf67, (2048, 768), (768, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), out=buf104)
        del arg68_1
        buf105 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), out=buf105)
        del arg70_1
        buf106 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf105, arg71_1, buf106, 1572864, grid=grid(1572864), stream=stream0)
        del arg71_1
        buf107 = reinterpret_tensor(buf105, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf105  # reuse
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf104, arg69_1, buf107, 1572864, grid=grid(1572864), stream=stream0)
        del arg69_1
        buf108 = buf88; del buf88  # reuse
        # Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf106, (12, 64, 2048), (131072, 1, 64), 0), out=buf108)
        buf113 = buf83; del buf83  # reuse
        # Source Nodes: [attn_weights_24], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf108, buf113, 24576, 2048, grid=grid(24576), stream=stream0)
        buf111 = reinterpret_tensor(buf107, (2048, 768), (768, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), out=buf111)
        del arg72_1
        buf112 = reinterpret_tensor(buf103, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf111, arg73_1, buf112, 1572864, grid=grid(1572864), stream=stream0)
        del arg73_1
        buf114 = reinterpret_tensor(buf111, (12, 2048, 64), (131072, 64, 1), 0); del buf111  # reuse
        # Source Nodes: [attn_output_20, attn_weights_24], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf113, reinterpret_tensor(buf112, (12, 2048, 64), (131072, 64, 1), 0), out=buf114)
        buf115 = reinterpret_tensor(buf104, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf104  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf114, buf115, 1572864, grid=grid(1572864), stream=stream0)
        buf116 = reinterpret_tensor(buf114, (2048, 768), (768, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (2048, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), out=buf116)
        del arg74_1
        buf117 = reinterpret_tensor(buf116, (1, 2048, 768), (1572864, 768, 1), 0); del buf116  # reuse
        buf121 = reinterpret_tensor(buf115, (2048, 768), (768, 1), 0); del buf115  # reuse
        # Source Nodes: [hidden_states_52, hidden_states_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf117, buf92, buf99, arg65_1, arg75_1, arg76_1, arg77_1, buf121, 2048, 768, grid=grid(2048), stream=stream0)
        del arg65_1
        del arg75_1
        del arg76_1
        del arg77_1
        buf122 = buf98; del buf98  # reuse
        # Source Nodes: [hidden_states_54], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf121, reinterpret_tensor(arg78_1, (768, 3072), (1, 768), 0), out=buf122)
        del arg78_1
        buf123 = buf122; del buf122  # reuse
        # Source Nodes: [hidden_states_56], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf123, arg79_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg79_1
        buf124 = buf121; del buf121  # reuse
        # Source Nodes: [hidden_states_56], Original ATen: [aten.relu]
        extern_kernels.mm(buf123, reinterpret_tensor(arg80_1, (3072, 768), (1, 3072), 0), out=buf124)
        del arg80_1
        buf128 = reinterpret_tensor(buf99, (1, 2048, 768), (1572864, 768, 1), 0); del buf99  # reuse
        # Source Nodes: [hidden_states_61], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf117, buf124, arg81_1, arg82_1, arg83_1, buf128, 2048, 768, grid=grid(2048), stream=stream0)
        del arg82_1
        del arg83_1
        buf129 = reinterpret_tensor(buf92, (2048, 768), (768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (2048, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 768), (1, 768), 0), out=buf129)
        del arg84_1
        buf130 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (2048, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), out=buf130)
        del arg86_1
        buf131 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf130, arg87_1, buf131, 1572864, grid=grid(1572864), stream=stream0)
        del arg87_1
        buf132 = reinterpret_tensor(buf130, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf130  # reuse
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf129, arg85_1, buf132, 1572864, grid=grid(1572864), stream=stream0)
        del arg85_1
        buf133 = buf113; del buf113  # reuse
        # Source Nodes: [attn_weights_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf131, (12, 64, 2048), (131072, 1, 64), 0), out=buf133)
        buf138 = buf108; del buf108  # reuse
        # Source Nodes: [attn_weights_29], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf133, buf138, 24576, 2048, grid=grid(24576), stream=stream0)
        buf136 = reinterpret_tensor(buf132, (2048, 768), (768, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (2048, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), out=buf136)
        del arg88_1
        buf137 = reinterpret_tensor(buf128, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf128  # reuse
        # Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf136, arg89_1, buf137, 1572864, grid=grid(1572864), stream=stream0)
        del arg89_1
        buf139 = reinterpret_tensor(buf136, (12, 2048, 64), (131072, 64, 1), 0); del buf136  # reuse
        # Source Nodes: [attn_output_25, attn_weights_29], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf138, reinterpret_tensor(buf137, (12, 2048, 64), (131072, 64, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf129, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf129  # reuse
        # Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf139, buf140, 1572864, grid=grid(1572864), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (2048, 768), (768, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (2048, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), out=buf141)
        del arg90_1
        buf142 = reinterpret_tensor(buf141, (1, 2048, 768), (1572864, 768, 1), 0); del buf141  # reuse
        buf146 = reinterpret_tensor(buf140, (2048, 768), (768, 1), 0); del buf140  # reuse
        # Source Nodes: [hidden_states_64, hidden_states_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf142, buf117, buf124, arg81_1, arg91_1, arg92_1, arg93_1, buf146, 2048, 768, grid=grid(2048), stream=stream0)
        del arg81_1
        del arg91_1
        del arg92_1
        del arg93_1
        buf147 = buf123; del buf123  # reuse
        # Source Nodes: [hidden_states_66], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf146, reinterpret_tensor(arg94_1, (768, 3072), (1, 768), 0), out=buf147)
        del arg94_1
        buf148 = buf147; del buf147  # reuse
        # Source Nodes: [hidden_states_68], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf148, arg95_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg95_1
        buf149 = buf146; del buf146  # reuse
        # Source Nodes: [hidden_states_68], Original ATen: [aten.relu]
        extern_kernels.mm(buf148, reinterpret_tensor(arg96_1, (3072, 768), (1, 3072), 0), out=buf149)
        del arg96_1
        buf153 = reinterpret_tensor(buf124, (1, 2048, 768), (1572864, 768, 1), 0); del buf124  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf142, buf149, arg97_1, arg98_1, arg99_1, buf153, 2048, 768, grid=grid(2048), stream=stream0)
        del arg98_1
        del arg99_1
        buf154 = reinterpret_tensor(buf117, (2048, 768), (768, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (2048, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 768), (1, 768), 0), out=buf154)
        del arg100_1
        buf155 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (2048, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), out=buf155)
        del arg102_1
        buf156 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf155, arg103_1, buf156, 1572864, grid=grid(1572864), stream=stream0)
        del arg103_1
        buf157 = reinterpret_tensor(buf155, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf155  # reuse
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf154, arg101_1, buf157, 1572864, grid=grid(1572864), stream=stream0)
        del arg101_1
        buf158 = buf138; del buf138  # reuse
        # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf156, (12, 64, 2048), (131072, 1, 64), 0), out=buf158)
        buf163 = buf133; del buf133  # reuse
        # Source Nodes: [attn_weights_34], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf158, buf163, 24576, 2048, grid=grid(24576), stream=stream0)
        buf161 = reinterpret_tensor(buf157, (2048, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (2048, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), out=buf161)
        del arg104_1
        buf162 = reinterpret_tensor(buf153, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf153  # reuse
        # Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf161, arg105_1, buf162, 1572864, grid=grid(1572864), stream=stream0)
        del arg105_1
        buf164 = reinterpret_tensor(buf161, (12, 2048, 64), (131072, 64, 1), 0); del buf161  # reuse
        # Source Nodes: [attn_output_30, attn_weights_34], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf163, reinterpret_tensor(buf162, (12, 2048, 64), (131072, 64, 1), 0), out=buf164)
        buf165 = reinterpret_tensor(buf154, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf154  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf164, buf165, 1572864, grid=grid(1572864), stream=stream0)
        buf166 = reinterpret_tensor(buf164, (2048, 768), (768, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (2048, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), out=buf166)
        del arg106_1
        buf167 = reinterpret_tensor(buf166, (1, 2048, 768), (1572864, 768, 1), 0); del buf166  # reuse
        buf171 = reinterpret_tensor(buf165, (2048, 768), (768, 1), 0); del buf165  # reuse
        # Source Nodes: [hidden_states_76, hidden_states_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf167, buf142, buf149, arg97_1, arg107_1, arg108_1, arg109_1, buf171, 2048, 768, grid=grid(2048), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg97_1
        buf172 = buf148; del buf148  # reuse
        # Source Nodes: [hidden_states_78], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf171, reinterpret_tensor(arg110_1, (768, 3072), (1, 768), 0), out=buf172)
        del arg110_1
        buf173 = buf172; del buf172  # reuse
        # Source Nodes: [hidden_states_80], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf173, arg111_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg111_1
        buf174 = buf171; del buf171  # reuse
        # Source Nodes: [hidden_states_80], Original ATen: [aten.relu]
        extern_kernels.mm(buf173, reinterpret_tensor(arg112_1, (3072, 768), (1, 3072), 0), out=buf174)
        del arg112_1
        buf178 = reinterpret_tensor(buf149, (1, 2048, 768), (1572864, 768, 1), 0); del buf149  # reuse
        # Source Nodes: [hidden_states_85], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf167, buf174, arg113_1, arg114_1, arg115_1, buf178, 2048, 768, grid=grid(2048), stream=stream0)
        del arg114_1
        del arg115_1
        buf179 = reinterpret_tensor(buf142, (2048, 768), (768, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (2048, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), out=buf179)
        del arg116_1
        buf180 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (2048, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), out=buf180)
        del arg118_1
        buf181 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf180, arg119_1, buf181, 1572864, grid=grid(1572864), stream=stream0)
        del arg119_1
        buf182 = reinterpret_tensor(buf180, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf180  # reuse
        # Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf179, arg117_1, buf182, 1572864, grid=grid(1572864), stream=stream0)
        del arg117_1
        buf183 = buf163; del buf163  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf181, (12, 64, 2048), (131072, 1, 64), 0), out=buf183)
        buf188 = buf158; del buf158  # reuse
        # Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf183, buf188, 24576, 2048, grid=grid(24576), stream=stream0)
        buf186 = reinterpret_tensor(buf182, (2048, 768), (768, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (2048, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 768), (1, 768), 0), out=buf186)
        del arg120_1
        buf187 = reinterpret_tensor(buf178, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf178  # reuse
        # Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf186, arg121_1, buf187, 1572864, grid=grid(1572864), stream=stream0)
        del arg121_1
        buf189 = reinterpret_tensor(buf186, (12, 2048, 64), (131072, 64, 1), 0); del buf186  # reuse
        # Source Nodes: [attn_output_35, attn_weights_39], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf188, reinterpret_tensor(buf187, (12, 2048, 64), (131072, 64, 1), 0), out=buf189)
        buf190 = reinterpret_tensor(buf179, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf179  # reuse
        # Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf189, buf190, 1572864, grid=grid(1572864), stream=stream0)
        buf191 = reinterpret_tensor(buf189, (2048, 768), (768, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (2048, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 768), (1, 768), 0), out=buf191)
        del arg122_1
        buf192 = reinterpret_tensor(buf191, (1, 2048, 768), (1572864, 768, 1), 0); del buf191  # reuse
        buf196 = reinterpret_tensor(buf190, (2048, 768), (768, 1), 0); del buf190  # reuse
        # Source Nodes: [hidden_states_88, hidden_states_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf192, buf167, buf174, arg113_1, arg123_1, arg124_1, arg125_1, buf196, 2048, 768, grid=grid(2048), stream=stream0)
        del arg113_1
        del arg123_1
        del arg124_1
        del arg125_1
        buf197 = buf173; del buf173  # reuse
        # Source Nodes: [hidden_states_90], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf196, reinterpret_tensor(arg126_1, (768, 3072), (1, 768), 0), out=buf197)
        del arg126_1
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [hidden_states_92], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf198, arg127_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg127_1
        buf199 = buf196; del buf196  # reuse
        # Source Nodes: [hidden_states_92], Original ATen: [aten.relu]
        extern_kernels.mm(buf198, reinterpret_tensor(arg128_1, (3072, 768), (1, 3072), 0), out=buf199)
        del arg128_1
        buf203 = reinterpret_tensor(buf174, (1, 2048, 768), (1572864, 768, 1), 0); del buf174  # reuse
        # Source Nodes: [hidden_states_97], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf192, buf199, arg129_1, arg130_1, arg131_1, buf203, 2048, 768, grid=grid(2048), stream=stream0)
        del arg130_1
        del arg131_1
        buf204 = reinterpret_tensor(buf167, (2048, 768), (768, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (2048, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 768), (1, 768), 0), out=buf204)
        del arg132_1
        buf205 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (2048, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), out=buf205)
        del arg134_1
        buf206 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf205, arg135_1, buf206, 1572864, grid=grid(1572864), stream=stream0)
        del arg135_1
        buf207 = reinterpret_tensor(buf205, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf205  # reuse
        # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf204, arg133_1, buf207, 1572864, grid=grid(1572864), stream=stream0)
        del arg133_1
        buf208 = buf188; del buf188  # reuse
        # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf206, (12, 64, 2048), (131072, 1, 64), 0), out=buf208)
        buf213 = buf183; del buf183  # reuse
        # Source Nodes: [attn_weights_44], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf208, buf213, 24576, 2048, grid=grid(24576), stream=stream0)
        buf211 = reinterpret_tensor(buf207, (2048, 768), (768, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (2048, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), out=buf211)
        del arg136_1
        buf212 = reinterpret_tensor(buf203, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf211, arg137_1, buf212, 1572864, grid=grid(1572864), stream=stream0)
        del arg137_1
        buf214 = reinterpret_tensor(buf211, (12, 2048, 64), (131072, 64, 1), 0); del buf211  # reuse
        # Source Nodes: [attn_output_40, attn_weights_44], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf213, reinterpret_tensor(buf212, (12, 2048, 64), (131072, 64, 1), 0), out=buf214)
        buf215 = reinterpret_tensor(buf204, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf204  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf214, buf215, 1572864, grid=grid(1572864), stream=stream0)
        buf216 = reinterpret_tensor(buf214, (2048, 768), (768, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (2048, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), out=buf216)
        del arg138_1
        buf217 = reinterpret_tensor(buf216, (1, 2048, 768), (1572864, 768, 1), 0); del buf216  # reuse
        buf221 = reinterpret_tensor(buf215, (2048, 768), (768, 1), 0); del buf215  # reuse
        # Source Nodes: [hidden_states_100, hidden_states_102], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf217, buf192, buf199, arg129_1, arg139_1, arg140_1, arg141_1, buf221, 2048, 768, grid=grid(2048), stream=stream0)
        del arg129_1
        del arg139_1
        del arg140_1
        del arg141_1
        buf222 = buf198; del buf198  # reuse
        # Source Nodes: [hidden_states_102], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf221, reinterpret_tensor(arg142_1, (768, 3072), (1, 768), 0), out=buf222)
        del arg142_1
        buf223 = buf222; del buf222  # reuse
        # Source Nodes: [hidden_states_104], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf223, arg143_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg143_1
        buf224 = buf221; del buf221  # reuse
        # Source Nodes: [hidden_states_104], Original ATen: [aten.relu]
        extern_kernels.mm(buf223, reinterpret_tensor(arg144_1, (3072, 768), (1, 3072), 0), out=buf224)
        del arg144_1
        buf228 = reinterpret_tensor(buf199, (1, 2048, 768), (1572864, 768, 1), 0); del buf199  # reuse
        # Source Nodes: [hidden_states_109], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf217, buf224, arg145_1, arg146_1, arg147_1, buf228, 2048, 768, grid=grid(2048), stream=stream0)
        del arg146_1
        del arg147_1
        buf229 = reinterpret_tensor(buf192, (2048, 768), (768, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (2048, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 768), (1, 768), 0), out=buf229)
        del arg148_1
        buf230 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (2048, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 768), (1, 768), 0), out=buf230)
        del arg150_1
        buf231 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf230, arg151_1, buf231, 1572864, grid=grid(1572864), stream=stream0)
        del arg151_1
        buf232 = reinterpret_tensor(buf230, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf230  # reuse
        # Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf229, arg149_1, buf232, 1572864, grid=grid(1572864), stream=stream0)
        del arg149_1
        buf233 = buf213; del buf213  # reuse
        # Source Nodes: [attn_weights_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf231, (12, 64, 2048), (131072, 1, 64), 0), out=buf233)
        buf238 = buf208; del buf208  # reuse
        # Source Nodes: [attn_weights_49], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf233, buf238, 24576, 2048, grid=grid(24576), stream=stream0)
        buf236 = reinterpret_tensor(buf232, (2048, 768), (768, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (2048, 768), (768, 1), 0), reinterpret_tensor(arg152_1, (768, 768), (1, 768), 0), out=buf236)
        del arg152_1
        buf237 = reinterpret_tensor(buf228, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf228  # reuse
        # Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf236, arg153_1, buf237, 1572864, grid=grid(1572864), stream=stream0)
        del arg153_1
        buf239 = reinterpret_tensor(buf236, (12, 2048, 64), (131072, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [attn_output_45, attn_weights_49], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf238, reinterpret_tensor(buf237, (12, 2048, 64), (131072, 64, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf229, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf229  # reuse
        # Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf239, buf240, 1572864, grid=grid(1572864), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (2048, 768), (768, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (2048, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), out=buf241)
        del arg154_1
        buf242 = reinterpret_tensor(buf241, (1, 2048, 768), (1572864, 768, 1), 0); del buf241  # reuse
        buf246 = reinterpret_tensor(buf240, (2048, 768), (768, 1), 0); del buf240  # reuse
        # Source Nodes: [hidden_states_112, hidden_states_114], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf242, buf217, buf224, arg145_1, arg155_1, arg156_1, arg157_1, buf246, 2048, 768, grid=grid(2048), stream=stream0)
        del arg145_1
        del arg155_1
        del arg156_1
        del arg157_1
        buf247 = buf223; del buf223  # reuse
        # Source Nodes: [hidden_states_114], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf246, reinterpret_tensor(arg158_1, (768, 3072), (1, 768), 0), out=buf247)
        del arg158_1
        buf248 = buf247; del buf247  # reuse
        # Source Nodes: [hidden_states_116], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf248, arg159_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg159_1
        buf249 = buf246; del buf246  # reuse
        # Source Nodes: [hidden_states_116], Original ATen: [aten.relu]
        extern_kernels.mm(buf248, reinterpret_tensor(arg160_1, (3072, 768), (1, 3072), 0), out=buf249)
        del arg160_1
        buf253 = reinterpret_tensor(buf224, (1, 2048, 768), (1572864, 768, 1), 0); del buf224  # reuse
        # Source Nodes: [hidden_states_121], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf242, buf249, arg161_1, arg162_1, arg163_1, buf253, 2048, 768, grid=grid(2048), stream=stream0)
        del arg162_1
        del arg163_1
        buf254 = reinterpret_tensor(buf217, (2048, 768), (768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (2048, 768), (768, 1), 0), reinterpret_tensor(arg164_1, (768, 768), (1, 768), 0), out=buf254)
        del arg164_1
        buf255 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (2048, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 768), (1, 768), 0), out=buf255)
        del arg166_1
        buf256 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf255, arg167_1, buf256, 1572864, grid=grid(1572864), stream=stream0)
        del arg167_1
        buf257 = reinterpret_tensor(buf255, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf255  # reuse
        # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf254, arg165_1, buf257, 1572864, grid=grid(1572864), stream=stream0)
        del arg165_1
        buf258 = buf238; del buf238  # reuse
        # Source Nodes: [attn_weights_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf256, (12, 64, 2048), (131072, 1, 64), 0), out=buf258)
        buf263 = buf233; del buf233  # reuse
        # Source Nodes: [attn_weights_54], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf258, buf263, 24576, 2048, grid=grid(24576), stream=stream0)
        buf261 = reinterpret_tensor(buf257, (2048, 768), (768, 1), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (2048, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 768), (1, 768), 0), out=buf261)
        del arg168_1
        buf262 = reinterpret_tensor(buf253, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf253  # reuse
        # Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf261, arg169_1, buf262, 1572864, grid=grid(1572864), stream=stream0)
        del arg169_1
        buf264 = reinterpret_tensor(buf261, (12, 2048, 64), (131072, 64, 1), 0); del buf261  # reuse
        # Source Nodes: [attn_output_50, attn_weights_54], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf263, reinterpret_tensor(buf262, (12, 2048, 64), (131072, 64, 1), 0), out=buf264)
        buf265 = reinterpret_tensor(buf254, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf254  # reuse
        # Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf264, buf265, 1572864, grid=grid(1572864), stream=stream0)
        buf266 = reinterpret_tensor(buf264, (2048, 768), (768, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (2048, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), out=buf266)
        del arg170_1
        buf267 = reinterpret_tensor(buf266, (1, 2048, 768), (1572864, 768, 1), 0); del buf266  # reuse
        buf271 = reinterpret_tensor(buf265, (2048, 768), (768, 1), 0); del buf265  # reuse
        # Source Nodes: [hidden_states_124, hidden_states_126], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf267, buf242, buf249, arg161_1, arg171_1, arg172_1, arg173_1, buf271, 2048, 768, grid=grid(2048), stream=stream0)
        del arg161_1
        del arg171_1
        del arg172_1
        del arg173_1
        buf272 = buf248; del buf248  # reuse
        # Source Nodes: [hidden_states_126], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf271, reinterpret_tensor(arg174_1, (768, 3072), (1, 768), 0), out=buf272)
        del arg174_1
        buf273 = buf272; del buf272  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf273, arg175_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg175_1
        buf274 = buf271; del buf271  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.relu]
        extern_kernels.mm(buf273, reinterpret_tensor(arg176_1, (3072, 768), (1, 3072), 0), out=buf274)
        del arg176_1
        buf278 = reinterpret_tensor(buf249, (1, 2048, 768), (1572864, 768, 1), 0); del buf249  # reuse
        # Source Nodes: [hidden_states_133], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf267, buf274, arg177_1, arg178_1, arg179_1, buf278, 2048, 768, grid=grid(2048), stream=stream0)
        del arg178_1
        del arg179_1
        buf279 = reinterpret_tensor(buf242, (2048, 768), (768, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (2048, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 768), (1, 768), 0), out=buf279)
        del arg180_1
        buf280 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (2048, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), out=buf280)
        del arg182_1
        buf281 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf280, arg183_1, buf281, 1572864, grid=grid(1572864), stream=stream0)
        del arg183_1
        buf282 = reinterpret_tensor(buf280, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf280  # reuse
        # Source Nodes: [contiguous_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf279, arg181_1, buf282, 1572864, grid=grid(1572864), stream=stream0)
        del arg181_1
        buf283 = buf263; del buf263  # reuse
        # Source Nodes: [attn_weights_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf282, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf281, (12, 64, 2048), (131072, 1, 64), 0), out=buf283)
        buf288 = buf258; del buf258  # reuse
        # Source Nodes: [attn_weights_59], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf283, buf288, 24576, 2048, grid=grid(24576), stream=stream0)
        del buf283
        buf286 = reinterpret_tensor(buf282, (2048, 768), (768, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (2048, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), out=buf286)
        del arg184_1
        buf287 = reinterpret_tensor(buf278, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf278  # reuse
        # Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf286, arg185_1, buf287, 1572864, grid=grid(1572864), stream=stream0)
        del arg185_1
        buf289 = reinterpret_tensor(buf286, (12, 2048, 64), (131072, 64, 1), 0); del buf286  # reuse
        # Source Nodes: [attn_output_55, attn_weights_59], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf288, reinterpret_tensor(buf287, (12, 2048, 64), (131072, 64, 1), 0), out=buf289)
        del buf288
        buf290 = reinterpret_tensor(buf279, (1, 2048, 12, 64), (1572864, 768, 64, 1), 0); del buf279  # reuse
        # Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf289, buf290, 1572864, grid=grid(1572864), stream=stream0)
        buf291 = reinterpret_tensor(buf289, (2048, 768), (768, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (2048, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), out=buf291)
        del arg186_1
        buf292 = reinterpret_tensor(buf291, (1, 2048, 768), (1572864, 768, 1), 0); del buf291  # reuse
        buf296 = reinterpret_tensor(buf290, (2048, 768), (768, 1), 0); del buf290  # reuse
        # Source Nodes: [hidden_states_136, hidden_states_138], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf292, buf267, buf274, arg177_1, arg187_1, arg188_1, arg189_1, buf296, 2048, 768, grid=grid(2048), stream=stream0)
        del arg177_1
        del arg187_1
        del arg188_1
        del arg189_1
        del buf267
        buf297 = buf273; del buf273  # reuse
        # Source Nodes: [hidden_states_138], Original ATen: [aten.native_layer_norm]
        extern_kernels.mm(buf296, reinterpret_tensor(arg190_1, (768, 3072), (1, 768), 0), out=buf297)
        del arg190_1
        buf298 = buf297; del buf297  # reuse
        # Source Nodes: [hidden_states_140], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf298, arg191_1, 6291456, grid=grid(6291456), stream=stream0)
        del arg191_1
        buf299 = buf296; del buf296  # reuse
        # Source Nodes: [hidden_states_140], Original ATen: [aten.relu]
        extern_kernels.mm(buf298, reinterpret_tensor(arg192_1, (3072, 768), (1, 3072), 0), out=buf299)
        del arg192_1
        del buf298
        buf303 = reinterpret_tensor(buf274, (1, 2048, 768), (1572864, 768, 1), 0); del buf274  # reuse
        # Source Nodes: [hidden_states_145], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_7.run(buf292, buf299, arg193_1, arg194_1, arg195_1, buf303, 2048, 768, grid=grid(2048), stream=stream0)
        del arg193_1
        del arg194_1
        del arg195_1
        del buf292
        del buf299
        buf304 = empty((2048, 50272), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___lm_head], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (2048, 768), (768, 1), 0), reinterpret_tensor(arg196_1, (768, 50272), (1, 768), 0), out=buf304)
        del arg196_1
        del buf303
        buf305 = empty_strided((2047, 1), (1, 2047), device='cuda', dtype=torch.float32)
        buf306 = empty_strided((2047, 1), (1, 2047), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_9.run(buf304, buf305, buf306, 2047, 50272, grid=grid(2047), stream=stream0)
        buf307 = empty((), device='cuda', dtype=torch.float32)
        buf309 = buf307; del buf307  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_10.run(buf309, arg198_1, buf304, buf305, buf306, 1, 2047, grid=grid(1), stream=stream0)
        del arg198_1
        return (buf309, reinterpret_tensor(buf304, (1, 2048, 50272), (102957056, 50272, 1), 0), buf6, buf12, buf31, buf37, buf56, buf62, buf81, buf87, buf106, buf112, buf131, buf137, buf156, buf162, buf181, buf187, buf206, buf212, buf231, buf237, buf256, buf262, buf281, buf287, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2050, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((50272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((50272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    arg198_1 = rand_strided((1, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
