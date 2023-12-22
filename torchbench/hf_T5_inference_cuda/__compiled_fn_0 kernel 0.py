
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


# kernel path: /tmp/torchinductor_youkaichao/57/c57lmsyvdrxdjv5sb3y73yvxplnvxpmf3oi7p2ru2vv3azcxif6i.py
# Source Nodes: [add_28, hidden_states_84, inputs_embeds_1, normed_hidden_states_6, pow_14, rsqrt_13, variance_13], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_28 => add_35
# hidden_states_84 => mul_32
# inputs_embeds_1 => embedding_2
# normed_hidden_states_6 => mul_33
# pow_14 => pow_14
# rsqrt_13 => rsqrt_13
# variance_13 => mean_13
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 32128
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp0 + 32128
        tmp11 = tmp0 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp0)
        tl.device_assert((0 <= tmp12) & (tmp12 < 32128), "index out of bounds: 0 <= tmp12 < 32128")
        tmp13 = tl.load(in_ptr1 + (r1 + (512*tmp12)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = 512.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-06
        tmp17 = tmp15 + tmp16
        tmp18 = tl.math.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp9 * tmp19
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp20, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/pf/cpff7xgynowigychxez5npcjcn6ryuoqge3gsgevvssnz5ruv3w6.py
# Source Nodes: [scores_12], Original ATen: [aten.clone]
# scores_12 => clone_51
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072) % 8
    x3 = (xindex // 1048576)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (1048576*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqassvb4d3yx3v2w44ulkz4nivs4ox5tlhspy3wi3kwqimzaenf.py
# Source Nodes: [scores_12], Original ATen: [aten.clone]
# scores_12 => clone_52
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (1048576*y1)), None)
    tl.store(out_ptr0 + (x2 + (2048*y3)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccf3mkktih6u45o4mdnmknqmggpo72qjha5levlih5jjcjnfccnt.py
# Source Nodes: [softmax_6], Original ATen: [aten._softmax]
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
triton_red_fused__softmax_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048) % 8
    _tmp33 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = (-1)*(tl.math.min(0, r3 + ((-1)*x0)))
        tmp2 = tl.full([1, 1], 16, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = tmp1.to(tl.float32)
        tmp5 = 16.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.log(tmp6)
        tmp8 = 2.0794415416798357
        tmp9 = tmp7 / tmp8
        tmp10 = tmp9 * tmp5
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tmp11 + tmp2
        tmp13 = tl.full([1, 1], 31, tl.int64)
        tmp14 = triton_helpers.minimum(tmp12, tmp13)
        tmp15 = tl.where(tmp3, tmp1, tmp14)
        tmp16 = tl.full([1, 1], 0, tl.int64)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17 + 32
        tmp19 = tmp17 < 0
        tmp20 = tl.where(tmp19, tmp18, tmp17)
        tl.device_assert((0 <= tmp20) & (tmp20 < 32), "index out of bounds: 0 <= tmp20 < 32")
        tmp21 = tl.load(in_ptr1 + (x1 + (8*tmp20)), None, eviction_policy='evict_last')
        tmp22 = r3
        tmp23 = x0
        tmp24 = tmp22 <= tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 1.0
        tmp27 = tmp26 - tmp25
        tmp28 = -3.4028234663852886e+38
        tmp29 = tmp27 * tmp28
        tmp30 = tmp21 + tmp29
        tmp31 = tmp0 + tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = triton_helpers.maximum(_tmp33, tmp32)
        _tmp33 = tl.where(rmask, tmp34, _tmp33)
    tmp33 = triton_helpers.max2(_tmp33, 1)[:, None]
    _tmp70 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp35 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = (-1)*(tl.math.min(0, r3 + ((-1)*x0)))
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp36.to(tl.float32)
        tmp40 = 16.0
        tmp41 = tmp39 / tmp40
        tmp42 = tl.log(tmp41)
        tmp43 = 2.0794415416798357
        tmp44 = tmp42 / tmp43
        tmp45 = tmp44 * tmp40
        tmp46 = tmp45.to(tl.int64)
        tmp47 = tmp46 + tmp37
        tmp48 = tl.full([1, 1], 31, tl.int64)
        tmp49 = triton_helpers.minimum(tmp47, tmp48)
        tmp50 = tl.where(tmp38, tmp36, tmp49)
        tmp51 = tl.full([1, 1], 0, tl.int64)
        tmp52 = tmp50 + tmp51
        tmp53 = tmp52 + 32
        tmp54 = tmp52 < 0
        tmp55 = tl.where(tmp54, tmp53, tmp52)
        tl.device_assert((0 <= tmp55) & (tmp55 < 32), "index out of bounds: 0 <= tmp55 < 32")
        tmp56 = tl.load(in_ptr1 + (x1 + (8*tmp55)), None, eviction_policy='evict_last')
        tmp57 = r3
        tmp58 = x0
        tmp59 = tmp57 <= tmp58
        tmp60 = tmp59.to(tl.float32)
        tmp61 = 1.0
        tmp62 = tmp61 - tmp60
        tmp63 = -3.4028234663852886e+38
        tmp64 = tmp62 * tmp63
        tmp65 = tmp56 + tmp64
        tmp66 = tmp35 + tmp65
        tmp67 = tmp66 - tmp33
        tmp68 = tl.exp(tmp67)
        tmp69 = tl.broadcast_to(tmp68, [XBLOCK, RBLOCK])
        tmp71 = _tmp70 + tmp69
        _tmp70 = tl.where(rmask, tmp71, _tmp70)
        tl.store(out_ptr1 + (r3 + (2048*x4)), tmp67, rmask)
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp72 = tl.load(out_ptr1 + (r3 + (2048*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.exp(tmp72)
        tmp74 = tmp73 / tmp70
        tl.store(out_ptr3 + (r3 + (2048*x4)), tmp74, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ue/cueskp26hniemwxxnt7ebwhaywjquidebtkrg3q4tzmfbpgvwbuz.py
# Source Nodes: [contiguous_6], Original ATen: [aten.clone]
# contiguous_6 => clone_55
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512) % 2048
    x3 = (xindex // 1048576)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (131072*x1) + (1048576*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinptf7ifc3nlc7dcnepjm5fx32ddvdmgegnvvx5pa6qt65ww4ow.py
# Source Nodes: [add, add_33, hidden_states_1, hidden_states_88, hidden_states_89, inputs_embeds, inputs_embeds_1, normed_hidden_states, normed_hidden_states_7, pow_1, pow_15, rsqrt, rsqrt_14, variance, variance_14], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add => add
# add_33 => add_41
# hidden_states_1 => mul_1
# hidden_states_88 => add_40
# hidden_states_89 => mul_35
# inputs_embeds => embedding
# inputs_embeds_1 => embedding_2
# normed_hidden_states => mul_2
# normed_hidden_states_7 => mul_36
# pow_1 => pow_1
# pow_15 => pow_15
# rsqrt => rsqrt
# rsqrt_14 => rsqrt_14
# variance => mean
# variance_14 => mean_14
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 32128
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp12 = tmp11 + 32128
        tmp13 = tmp11 < 0
        tmp14 = tl.where(tmp13, tmp12, tmp11)
        tl.device_assert((0 <= tmp14) & (tmp14 < 32128), "index out of bounds: 0 <= tmp14 < 32128")
        tmp15 = tl.load(in_ptr1 + (r1 + (512*tmp14)), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp0 + 32128
        tmp22 = tmp0 < 0
        tmp23 = tl.where(tmp22, tmp21, tmp0)
        tl.device_assert((0 <= tmp23) & (tmp23 < 32128), "index out of bounds: 0 <= tmp23 < 32128")
        tmp24 = tl.load(in_ptr1 + (r1 + (512*tmp23)), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tmp24 + tmp25
        tmp27 = 512.0
        tmp28 = tmp9 / tmp27
        tmp29 = 1e-06
        tmp30 = tmp28 + tmp29
        tmp31 = tl.math.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp33 = tmp20 * tmp32
        tmp35 = tmp11 + 32128
        tmp36 = tmp11 < 0
        tmp37 = tl.where(tmp36, tmp35, tmp11)
        tl.device_assert((0 <= tmp37) & (tmp37 < 32128), "index out of bounds: 0 <= tmp37 < 32128")
        tmp38 = tl.load(in_ptr1 + (r1 + (512*tmp37)), rmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tmp18 / tmp27
        tmp40 = tmp39 + tmp29
        tmp41 = tl.math.rsqrt(tmp40)
        tmp42 = tmp38 * tmp41
        tmp43 = tmp34 * tmp42
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp33, rmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp43, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvlhrq2g35ouja55wpvh6lw7wqsw6gza7afxvddht6g3ojdukqg.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, div_2, exp, sub_2, sum_1
triton_red_fused__softmax_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048) % 8
    _tmp32 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r3 + ((-1)*x0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 > tmp2
        tmp4 = tmp3.to(tl.int64)
        tmp5 = tl.full([1, 1], 16, tl.int64)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6 + tmp2
        tmp8 = tl.abs(tmp1)
        tmp9 = tl.full([1, 1], 8, tl.int64)
        tmp10 = tmp8 < tmp9
        tmp11 = tmp8.to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = tl.log(tmp13)
        tmp15 = 2.772588722239781
        tmp16 = tmp14 / tmp15
        tmp17 = tmp16 * tmp12
        tmp18 = tmp17.to(tl.int64)
        tmp19 = tmp18 + tmp9
        tmp20 = tl.full([1, 1], 15, tl.int64)
        tmp21 = triton_helpers.minimum(tmp19, tmp20)
        tmp22 = tl.where(tmp10, tmp8, tmp21)
        tmp23 = tmp7 + tmp22
        tmp24 = tmp23 + 32
        tmp25 = tmp23 < 0
        tmp26 = tl.where(tmp25, tmp24, tmp23)
        tl.device_assert(((0 <= tmp26) & (tmp26 < 32)) | ~rmask, "index out of bounds: 0 <= tmp26 < 32")
        tmp27 = tl.load(in_ptr1 + (x1 + (8*tmp26)), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = 0.0
        tmp29 = tmp27 + tmp28
        tmp30 = tmp0 + tmp29
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = triton_helpers.maximum(_tmp32, tmp31)
        _tmp32 = tl.where(rmask, tmp33, _tmp32)
    tmp32 = triton_helpers.max2(_tmp32, 1)[:, None]
    _tmp68 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp34 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = r3 + ((-1)*x0)
        tmp36 = tl.full([1, 1], 0, tl.int64)
        tmp37 = tmp35 > tmp36
        tmp38 = tmp37.to(tl.int64)
        tmp39 = tl.full([1, 1], 16, tl.int64)
        tmp40 = tmp38 * tmp39
        tmp41 = tmp40 + tmp36
        tmp42 = tl.abs(tmp35)
        tmp43 = tl.full([1, 1], 8, tl.int64)
        tmp44 = tmp42 < tmp43
        tmp45 = tmp42.to(tl.float32)
        tmp46 = 8.0
        tmp47 = tmp45 / tmp46
        tmp48 = tl.log(tmp47)
        tmp49 = 2.772588722239781
        tmp50 = tmp48 / tmp49
        tmp51 = tmp50 * tmp46
        tmp52 = tmp51.to(tl.int64)
        tmp53 = tmp52 + tmp43
        tmp54 = tl.full([1, 1], 15, tl.int64)
        tmp55 = triton_helpers.minimum(tmp53, tmp54)
        tmp56 = tl.where(tmp44, tmp42, tmp55)
        tmp57 = tmp41 + tmp56
        tmp58 = tmp57 + 32
        tmp59 = tmp57 < 0
        tmp60 = tl.where(tmp59, tmp58, tmp57)
        tl.device_assert(((0 <= tmp60) & (tmp60 < 32)) | ~rmask, "index out of bounds: 0 <= tmp60 < 32")
        tmp61 = tl.load(in_ptr1 + (x1 + (8*tmp60)), rmask, eviction_policy='evict_last', other=0.0)
        tmp62 = 0.0
        tmp63 = tmp61 + tmp62
        tmp64 = tmp34 + tmp63
        tmp65 = tmp64 - tmp32
        tmp66 = tl.exp(tmp65)
        tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
        tmp69 = _tmp68 + tmp67
        _tmp68 = tl.where(rmask, tmp69, _tmp68)
    tmp68 = tl.sum(_tmp68, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp70 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp71 = r3 + ((-1)*x0)
        tmp72 = tl.full([1, 1], 0, tl.int64)
        tmp73 = tmp71 > tmp72
        tmp74 = tmp73.to(tl.int64)
        tmp75 = tl.full([1, 1], 16, tl.int64)
        tmp76 = tmp74 * tmp75
        tmp77 = tmp76 + tmp72
        tmp78 = tl.abs(tmp71)
        tmp79 = tl.full([1, 1], 8, tl.int64)
        tmp80 = tmp78 < tmp79
        tmp81 = tmp78.to(tl.float32)
        tmp82 = 8.0
        tmp83 = tmp81 / tmp82
        tmp84 = tl.log(tmp83)
        tmp85 = 2.772588722239781
        tmp86 = tmp84 / tmp85
        tmp87 = tmp86 * tmp82
        tmp88 = tmp87.to(tl.int64)
        tmp89 = tmp88 + tmp79
        tmp90 = tl.full([1, 1], 15, tl.int64)
        tmp91 = triton_helpers.minimum(tmp89, tmp90)
        tmp92 = tl.where(tmp80, tmp78, tmp91)
        tmp93 = tmp77 + tmp92
        tmp94 = tmp93 + 32
        tmp95 = tmp93 < 0
        tmp96 = tl.where(tmp95, tmp94, tmp93)
        tl.device_assert(((0 <= tmp96) & (tmp96 < 32)) | ~rmask, "index out of bounds: 0 <= tmp96 < 32")
        tmp97 = tl.load(in_ptr1 + (x1 + (8*tmp96)), rmask, eviction_policy='evict_last', other=0.0)
        tmp98 = 0.0
        tmp99 = tmp97 + tmp98
        tmp100 = tmp70 + tmp99
        tmp101 = tmp100 - tmp32
        tmp102 = tl.exp(tmp101)
        tmp103 = tmp102 / tmp68
        tl.store(out_ptr2 + (r3 + (2048*x4)), tmp103, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mfb5l636mxm7tjwg4wdug2mopve3a3uvczeykawsycwjs65gkt.py
# Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, inputs_embeds, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_5 => add_7
# forwarded_states => mul_6
# hidden_states_5 => add_6
# hidden_states_6 => mul_5
# inputs_embeds => embedding
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# variance_1 => mean_1
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 32128
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp0 + 32128
        tmp13 = tmp0 < 0
        tmp14 = tl.where(tmp13, tmp12, tmp0)
        tl.device_assert((0 <= tmp14) & (tmp14 < 32128), "index out of bounds: 0 <= tmp14 < 32128")
        tmp15 = tl.load(in_ptr1 + (r1 + (512*tmp14)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp18 = 512.0
        tmp19 = tmp9 / tmp18
        tmp20 = 1e-06
        tmp21 = tmp19 + tmp20
        tmp22 = tl.math.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp24 = tmp11 * tmp23
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp24, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cumj4mvw4otwr4coc3txshavhtpft4l7gbqttx6fpmd467mi56pd.py
# Source Nodes: [hidden_states_8], Original ATen: [aten.relu]
# hidden_states_8 => relu
triton_poi_fused_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexeauswm6k2lrjbqbecmevuore6jlcys2muwudcnpk2zumxpjhy.py
# Source Nodes: [add_7, hidden_states_13, hidden_states_14, hidden_states_5, inputs_embeds, normed_hidden_states_1, pow_3, rsqrt_2, variance_2], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_7 => add_9
# hidden_states_13 => add_8
# hidden_states_14 => mul_7
# hidden_states_5 => add_6
# inputs_embeds => embedding
# normed_hidden_states_1 => mul_8
# pow_3 => pow_3
# rsqrt_2 => rsqrt_2
# variance_2 => mean_2
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 32128
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp0 + 32128
        tmp15 = tmp0 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp0)
        tl.device_assert((0 <= tmp16) & (tmp16 < 32128), "index out of bounds: 0 <= tmp16 < 32128")
        tmp17 = tl.load(in_ptr1 + (r1 + (512*tmp16)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp21 = tmp19 + tmp20
        tmp22 = 512.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-06
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp28 = tmp13 * tmp27
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp28, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nz/cnz4neig5k5iode3vqdpnbsrjte4dsywgdymj2lrxdwbvmhbe7ud.py
# Source Nodes: [add_9, forwarded_states_2, hidden_states_13, hidden_states_18, hidden_states_19, hidden_states_5, inputs_embeds, pow_4, rsqrt_3, variance_3], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_9 => add_12
# forwarded_states_2 => mul_10
# hidden_states_13 => add_8
# hidden_states_18 => add_11
# hidden_states_19 => mul_9
# hidden_states_5 => add_6
# inputs_embeds => embedding
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
# variance_3 => mean_3
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 32128
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
    tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 512.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp10 * tmp21
    tmp23 = tmp16 * tmp22
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp10, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp23, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cneqzxo6qmudike2m6tin6apyfxut3hvwgxronz7lnx7bujxyrrw.py
# Source Nodes: [add_11, hidden_states_26, hidden_states_27, normed_hidden_states_2, pow_5, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_11 => add_14
# hidden_states_26 => add_13
# hidden_states_27 => mul_11
# normed_hidden_states_2 => mul_12
# pow_5 => pow_5
# rsqrt_4 => rsqrt_4
# variance_4 => mean_4
triton_per_fused_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 8192
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = 512.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = tmp2 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3funcpdkhwptufwwfp5rmwhob545wkuhviduxk4vynipksv6l4.py
# Source Nodes: [add_13, forwarded_states_4, hidden_states_26, hidden_states_31, hidden_states_32, pow_6, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_13 => add_17
# forwarded_states_4 => mul_14
# hidden_states_26 => add_13
# hidden_states_31 => add_16
# hidden_states_32 => mul_13
# pow_6 => pow_6
# rsqrt_5 => rsqrt_5
# variance_5 => mean_5
triton_per_fused_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 8192
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 512.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp16 = tmp4 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp17, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3run5iufikha6mtoqsykn2ilqaai5rncsxyq2cwvsc35t5wtmi5.py
# Source Nodes: [add_15, hidden_states_26, hidden_states_31, hidden_states_39, hidden_states_40, normed_hidden_states_3, pow_7, rsqrt_6, variance_6], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_15 => add_19
# hidden_states_26 => add_13
# hidden_states_31 => add_16
# hidden_states_39 => add_18
# hidden_states_40 => mul_15
# normed_hidden_states_3 => mul_16
# pow_7 => pow_7
# rsqrt_6 => rsqrt_6
# variance_6 => mean_6
triton_per_fused_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 8192
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.rsqrt(tmp16)
    tmp18 = tmp6 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc55gnmdvueywgzg6otinu2uuo3y5voks32orluzqznpdpfbedxe.py
# Source Nodes: [add_17, forwarded_states_6, hidden_states_26, hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_45, pow_8, rsqrt_7, variance_7], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_17 => add_22
# forwarded_states_6 => mul_18
# hidden_states_26 => add_13
# hidden_states_31 => add_16
# hidden_states_39 => add_18
# hidden_states_44 => add_21
# hidden_states_45 => mul_17
# pow_8 => pow_8
# rsqrt_7 => rsqrt_7
# variance_7 => mean_7
triton_per_fused_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 8192
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3ohn4nbrfiurhwvefyzlexwgkmgwxehprmv3wfbbafo7a62rrh.py
# Source Nodes: [add_25, forwarded_states_10, hidden_states_52, hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_71, pow_12, rsqrt_11, variance_11], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_25 => add_32
# forwarded_states_10 => mul_26
# hidden_states_52 => add_23
# hidden_states_57 => add_26
# hidden_states_65 => add_28
# hidden_states_70 => add_31
# hidden_states_71 => mul_25
# pow_12 => pow_12
# rsqrt_11 => rsqrt_11
# variance_11 => mean_11
triton_per_fused_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 8192
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
    tmp1 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/ccotde7cvfpguwww36gywcqsplmixw7svi55xj5jkg7pgjfprv6q.py
# Source Nodes: [softmax_7], Original ATen: [aten._softmax]
# softmax_7 => amax_7, div_11, exp_7, sub_12, sum_8
triton_red_fused__softmax_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 2048
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
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.exp(tmp11)
        tmp13 = tmp12 / tmp8
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmfdy7uymlhu2ix55vf25qfmyomeudo4ja5oxmk6lcbb7hqtxtke.py
# Source Nodes: [add_68, hidden_states_173, hidden_states_177, hidden_states_185, hidden_states_186, hidden_states_187, pow_32, rsqrt_31, sequence_output_1, variance_31], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_68 => add_87
# hidden_states_173 => add_81
# hidden_states_177 => add_84
# hidden_states_185 => add_86
# hidden_states_186 => mul_69
# hidden_states_187 => mul_70
# pow_32 => pow_32
# rsqrt_31 => rsqrt_31
# sequence_output_1 => mul_71
# variance_31 => mean_31
triton_per_fused_add_mean_mul_pow_rsqrt_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 8192
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
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.rsqrt(tmp16)
    tmp18 = tmp6 * tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = 0.04419417382415922
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, ), (1, ))
    assert_size_stride(arg1_1, (512, ), (1, ))
    assert_size_stride(arg2_1, (512, ), (1, ))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, ), (1, ))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (32128, 512), (512, 1))
    assert_size_stride(arg33_1, (512, 512), (512, 1))
    assert_size_stride(arg34_1, (512, 512), (512, 1))
    assert_size_stride(arg35_1, (512, 512), (512, 1))
    assert_size_stride(arg36_1, (32, 8), (8, 1))
    assert_size_stride(arg37_1, (512, 512), (512, 1))
    assert_size_stride(arg38_1, (2048, 512), (512, 1))
    assert_size_stride(arg39_1, (512, 2048), (2048, 1))
    assert_size_stride(arg40_1, (512, 512), (512, 1))
    assert_size_stride(arg41_1, (512, 512), (512, 1))
    assert_size_stride(arg42_1, (512, 512), (512, 1))
    assert_size_stride(arg43_1, (512, 512), (512, 1))
    assert_size_stride(arg44_1, (2048, 512), (512, 1))
    assert_size_stride(arg45_1, (512, 2048), (2048, 1))
    assert_size_stride(arg46_1, (512, 512), (512, 1))
    assert_size_stride(arg47_1, (512, 512), (512, 1))
    assert_size_stride(arg48_1, (512, 512), (512, 1))
    assert_size_stride(arg49_1, (512, 512), (512, 1))
    assert_size_stride(arg50_1, (2048, 512), (512, 1))
    assert_size_stride(arg51_1, (512, 2048), (2048, 1))
    assert_size_stride(arg52_1, (512, 512), (512, 1))
    assert_size_stride(arg53_1, (512, 512), (512, 1))
    assert_size_stride(arg54_1, (512, 512), (512, 1))
    assert_size_stride(arg55_1, (512, 512), (512, 1))
    assert_size_stride(arg56_1, (2048, 512), (512, 1))
    assert_size_stride(arg57_1, (512, 2048), (2048, 1))
    assert_size_stride(arg58_1, (512, 512), (512, 1))
    assert_size_stride(arg59_1, (512, 512), (512, 1))
    assert_size_stride(arg60_1, (512, 512), (512, 1))
    assert_size_stride(arg61_1, (512, 512), (512, 1))
    assert_size_stride(arg62_1, (2048, 512), (512, 1))
    assert_size_stride(arg63_1, (512, 2048), (2048, 1))
    assert_size_stride(arg64_1, (512, 512), (512, 1))
    assert_size_stride(arg65_1, (512, 512), (512, 1))
    assert_size_stride(arg66_1, (512, 512), (512, 1))
    assert_size_stride(arg67_1, (512, 512), (512, 1))
    assert_size_stride(arg68_1, (2048, 512), (512, 1))
    assert_size_stride(arg69_1, (512, 2048), (2048, 1))
    assert_size_stride(arg70_1, (512, 512), (512, 1))
    assert_size_stride(arg71_1, (512, 512), (512, 1))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (32, 8), (8, 1))
    assert_size_stride(arg74_1, (512, 512), (512, 1))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg76_1, (512, 512), (512, 1))
    assert_size_stride(arg77_1, (512, 512), (512, 1))
    assert_size_stride(arg78_1, (512, 512), (512, 1))
    assert_size_stride(arg79_1, (2048, 512), (512, 1))
    assert_size_stride(arg80_1, (512, 2048), (2048, 1))
    assert_size_stride(arg81_1, (512, 512), (512, 1))
    assert_size_stride(arg82_1, (512, 512), (512, 1))
    assert_size_stride(arg83_1, (512, 512), (512, 1))
    assert_size_stride(arg84_1, (512, 512), (512, 1))
    assert_size_stride(arg85_1, (512, 512), (512, 1))
    assert_size_stride(arg86_1, (512, 512), (512, 1))
    assert_size_stride(arg87_1, (512, 512), (512, 1))
    assert_size_stride(arg88_1, (512, 512), (512, 1))
    assert_size_stride(arg89_1, (2048, 512), (512, 1))
    assert_size_stride(arg90_1, (512, 2048), (2048, 1))
    assert_size_stride(arg91_1, (512, 512), (512, 1))
    assert_size_stride(arg92_1, (512, 512), (512, 1))
    assert_size_stride(arg93_1, (512, 512), (512, 1))
    assert_size_stride(arg94_1, (512, 512), (512, 1))
    assert_size_stride(arg95_1, (512, 512), (512, 1))
    assert_size_stride(arg96_1, (512, 512), (512, 1))
    assert_size_stride(arg97_1, (512, 512), (512, 1))
    assert_size_stride(arg98_1, (512, 512), (512, 1))
    assert_size_stride(arg99_1, (2048, 512), (512, 1))
    assert_size_stride(arg100_1, (512, 2048), (2048, 1))
    assert_size_stride(arg101_1, (512, 512), (512, 1))
    assert_size_stride(arg102_1, (512, 512), (512, 1))
    assert_size_stride(arg103_1, (512, 512), (512, 1))
    assert_size_stride(arg104_1, (512, 512), (512, 1))
    assert_size_stride(arg105_1, (512, 512), (512, 1))
    assert_size_stride(arg106_1, (512, 512), (512, 1))
    assert_size_stride(arg107_1, (512, 512), (512, 1))
    assert_size_stride(arg108_1, (512, 512), (512, 1))
    assert_size_stride(arg109_1, (2048, 512), (512, 1))
    assert_size_stride(arg110_1, (512, 2048), (2048, 1))
    assert_size_stride(arg111_1, (512, 512), (512, 1))
    assert_size_stride(arg112_1, (512, 512), (512, 1))
    assert_size_stride(arg113_1, (512, 512), (512, 1))
    assert_size_stride(arg114_1, (512, 512), (512, 1))
    assert_size_stride(arg115_1, (512, 512), (512, 1))
    assert_size_stride(arg116_1, (512, 512), (512, 1))
    assert_size_stride(arg117_1, (512, 512), (512, 1))
    assert_size_stride(arg118_1, (512, 512), (512, 1))
    assert_size_stride(arg119_1, (2048, 512), (512, 1))
    assert_size_stride(arg120_1, (512, 2048), (2048, 1))
    assert_size_stride(arg121_1, (512, 512), (512, 1))
    assert_size_stride(arg122_1, (512, 512), (512, 1))
    assert_size_stride(arg123_1, (512, 512), (512, 1))
    assert_size_stride(arg124_1, (512, 512), (512, 1))
    assert_size_stride(arg125_1, (512, 512), (512, 1))
    assert_size_stride(arg126_1, (512, 512), (512, 1))
    assert_size_stride(arg127_1, (512, 512), (512, 1))
    assert_size_stride(arg128_1, (512, 512), (512, 1))
    assert_size_stride(arg129_1, (2048, 512), (512, 1))
    assert_size_stride(arg130_1, (512, 2048), (2048, 1))
    assert_size_stride(arg131_1, (32128, 512), (512, 1))
    assert_size_stride(arg132_1, (4, 2048), (2048, 1))
    assert_size_stride(arg133_1, (4, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty((4, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, hidden_states_84, inputs_embeds_1, normed_hidden_states_6, pow_14, rsqrt_13, variance_13], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0.run(arg133_1, arg32_1, arg13_1, buf1, 8192, 512, grid=grid(8192), stream=stream0)
        del arg13_1
        buf2 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (8192, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 512), (1, 512), 0), out=buf2)
        del arg70_1
        buf3 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (8192, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 512), (1, 512), 0), out=buf3)
        del arg71_1
        buf4 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf2, buf4, 4194304, grid=grid(4194304), stream=stream0)
        buf5 = reinterpret_tensor(buf2, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf2  # reuse
        # Source Nodes: [scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf3, buf5, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf6 = empty((32, 2048, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf5, (32, 64, 2048), (131072, 2048, 1), 0), out=buf6)
        buf8 = empty((4, 8, 2048, 2048), device='cuda', dtype=torch.float32)
        buf11 = empty((4, 8, 2048, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_6], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf6, arg73_1, buf8, buf11, 65536, 2048, grid=grid(65536), stream=stream0)
        buf10 = reinterpret_tensor(buf5, (8192, 512), (512, 1), 0); del buf5  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (8192, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), out=buf10)
        del arg72_1
        buf12 = reinterpret_tensor(buf1, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf1  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf10, buf12, 4194304, grid=grid(4194304), stream=stream0)
        buf13 = reinterpret_tensor(buf4, (32, 2048, 64), (131072, 64, 1), 0); del buf4  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf12, (32, 2048, 64), (131072, 64, 1), 0), out=buf13)
        buf14 = reinterpret_tensor(buf12, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf12  # reuse
        # Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf13, buf14, 4194304, grid=grid(4194304), stream=stream0)
        buf15 = reinterpret_tensor(buf13, (8192, 512), (512, 1), 0); del buf13  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (8192, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf15)
        del arg74_1
        buf17 = reinterpret_tensor(buf14, (4, 2048, 512), (1048576, 512, 1), 0); del buf14  # reuse
        buf20 = empty((4, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_33, hidden_states_1, hidden_states_88, hidden_states_89, inputs_embeds, inputs_embeds_1, normed_hidden_states, normed_hidden_states_7, pow_1, pow_15, rsqrt, rsqrt_14, variance, variance_14], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_5.run(arg133_1, arg32_1, buf15, arg132_1, arg14_1, arg0_1, buf17, buf20, 8192, 512, grid=grid(8192), stream=stream0)
        del arg0_1
        del arg14_1
        buf18 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (8192, 512), (512, 1), 0), reinterpret_tensor(arg75_1, (512, 512), (1, 512), 0), out=buf18)
        del arg75_1
        buf21 = reinterpret_tensor(buf17, (8192, 512), (512, 1), 0); del buf17  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (8192, 512), (512, 1), 0), reinterpret_tensor(arg33_1, (512, 512), (1, 512), 0), out=buf21)
        del arg33_1
        buf22 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (8192, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 512), (1, 512), 0), out=buf22)
        del arg34_1
        buf23 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf21, buf23, 4194304, grid=grid(4194304), stream=stream0)
        buf24 = reinterpret_tensor(buf21, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf21  # reuse
        # Source Nodes: [scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf22, buf24, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf25 = reinterpret_tensor(buf11, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf11  # reuse
        # Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf24, (32, 64, 2048), (131072, 2048, 1), 0), out=buf25)
        buf28 = buf8; del buf8  # reuse
        # Source Nodes: [softmax], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf25, arg36_1, buf28, 65536, 2048, grid=grid(65536), stream=stream0)
        buf29 = reinterpret_tensor(buf24, (8192, 512), (512, 1), 0); del buf24  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (8192, 512), (512, 1), 0), reinterpret_tensor(arg35_1, (512, 512), (1, 512), 0), out=buf29)
        del arg35_1
        buf30 = reinterpret_tensor(buf20, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf29, buf30, 4194304, grid=grid(4194304), stream=stream0)
        buf31 = reinterpret_tensor(buf29, (32, 2048, 64), (131072, 64, 1), 0); del buf29  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf30, (32, 2048, 64), (131072, 64, 1), 0), out=buf31)
        buf32 = reinterpret_tensor(buf30, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf30  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf31, buf32, 4194304, grid=grid(4194304), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (8192, 512), (512, 1), 0); del buf31  # reuse
        # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (8192, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 512), (1, 512), 0), out=buf33)
        del arg37_1
        buf35 = reinterpret_tensor(buf32, (4, 2048, 512), (1048576, 512, 1), 0); del buf32  # reuse
        # Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, inputs_embeds, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_7.run(arg132_1, arg32_1, buf33, arg1_1, buf35, 8192, 512, grid=grid(8192), stream=stream0)
        del arg1_1
        buf36 = empty((8192, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (8192, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 2048), (1, 512), 0), out=buf36)
        del arg38_1
        buf37 = reinterpret_tensor(buf36, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf36  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf37, 16777216, grid=grid(16777216), stream=stream0)
        buf38 = reinterpret_tensor(buf35, (8192, 512), (512, 1), 0); del buf35  # reuse
        # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg39_1, (2048, 512), (1, 2048), 0), out=buf38)
        del arg39_1
        buf40 = reinterpret_tensor(buf23, (4, 2048, 512), (1048576, 512, 1), 0); del buf23  # reuse
        # Source Nodes: [add_7, hidden_states_13, hidden_states_14, hidden_states_5, inputs_embeds, normed_hidden_states_1, pow_3, rsqrt_2, variance_2], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_9.run(arg132_1, arg32_1, buf33, buf38, arg2_1, buf40, 8192, 512, grid=grid(8192), stream=stream0)
        del arg2_1
        buf41 = buf22; del buf22  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (8192, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 512), (1, 512), 0), out=buf41)
        del arg40_1
        buf42 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (8192, 512), (512, 1), 0), reinterpret_tensor(arg41_1, (512, 512), (1, 512), 0), out=buf42)
        del arg41_1
        buf43 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf41, buf43, 4194304, grid=grid(4194304), stream=stream0)
        buf44 = reinterpret_tensor(buf41, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf41  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf42, buf44, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf45 = reinterpret_tensor(buf28, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf28  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf44, (32, 64, 2048), (131072, 2048, 1), 0), out=buf45)
        buf48 = reinterpret_tensor(buf25, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf25  # reuse
        # Source Nodes: [softmax_1], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf45, arg36_1, buf48, 65536, 2048, grid=grid(65536), stream=stream0)
        buf49 = reinterpret_tensor(buf44, (8192, 512), (512, 1), 0); del buf44  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (8192, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 512), (1, 512), 0), out=buf49)
        del arg42_1
        buf50 = reinterpret_tensor(buf40, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf40  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf49, buf50, 4194304, grid=grid(4194304), stream=stream0)
        buf51 = reinterpret_tensor(buf49, (32, 2048, 64), (131072, 64, 1), 0); del buf49  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf50, (32, 2048, 64), (131072, 64, 1), 0), out=buf51)
        buf52 = reinterpret_tensor(buf50, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf50  # reuse
        # Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf51, buf52, 4194304, grid=grid(4194304), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (8192, 512), (512, 1), 0); del buf51  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (8192, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 512), (1, 512), 0), out=buf53)
        del arg43_1
        buf54 = reinterpret_tensor(buf33, (4, 2048, 512), (1048576, 512, 1), 0); del buf33  # reuse
        buf56 = reinterpret_tensor(buf52, (4, 2048, 512), (1048576, 512, 1), 0); del buf52  # reuse
        # Source Nodes: [add_9, forwarded_states_2, hidden_states_13, hidden_states_18, hidden_states_19, hidden_states_5, inputs_embeds, pow_4, rsqrt_3, variance_3], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10.run(buf54, arg132_1, arg32_1, buf38, buf53, arg3_1, buf56, 8192, 512, grid=grid(8192), stream=stream0)
        del arg132_1
        del arg3_1
        buf57 = reinterpret_tensor(buf37, (8192, 2048), (2048, 1), 0); del buf37  # reuse
        # Source Nodes: [hidden_states_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (8192, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 2048), (1, 512), 0), out=buf57)
        del arg44_1
        buf58 = reinterpret_tensor(buf57, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf57  # reuse
        # Source Nodes: [hidden_states_21], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf58, 16777216, grid=grid(16777216), stream=stream0)
        buf59 = reinterpret_tensor(buf56, (8192, 512), (512, 1), 0); del buf56  # reuse
        # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg45_1, (2048, 512), (1, 2048), 0), out=buf59)
        del arg45_1
        buf61 = reinterpret_tensor(buf53, (4, 2048, 512), (1048576, 512, 1), 0); del buf53  # reuse
        # Source Nodes: [add_11, hidden_states_26, hidden_states_27, normed_hidden_states_2, pow_5, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf54, buf59, arg4_1, buf61, 8192, 512, grid=grid(8192), stream=stream0)
        del arg4_1
        buf62 = buf38; del buf38  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (8192, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 512), (1, 512), 0), out=buf62)
        del arg46_1
        buf63 = reinterpret_tensor(buf43, (8192, 512), (512, 1), 0); del buf43  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (8192, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 512), (1, 512), 0), out=buf63)
        del arg47_1
        buf64 = reinterpret_tensor(buf42, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf42  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf62, buf64, 4194304, grid=grid(4194304), stream=stream0)
        buf65 = reinterpret_tensor(buf62, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf62  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf63, buf65, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf66 = reinterpret_tensor(buf48, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf48  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf65, (32, 64, 2048), (131072, 2048, 1), 0), out=buf66)
        buf69 = reinterpret_tensor(buf45, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf45  # reuse
        # Source Nodes: [softmax_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf66, arg36_1, buf69, 65536, 2048, grid=grid(65536), stream=stream0)
        buf70 = reinterpret_tensor(buf65, (8192, 512), (512, 1), 0); del buf65  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (8192, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 512), (1, 512), 0), out=buf70)
        del arg48_1
        buf71 = reinterpret_tensor(buf61, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf61  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf70, buf71, 4194304, grid=grid(4194304), stream=stream0)
        buf72 = reinterpret_tensor(buf70, (32, 2048, 64), (131072, 64, 1), 0); del buf70  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf69, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf71, (32, 2048, 64), (131072, 64, 1), 0), out=buf72)
        buf73 = reinterpret_tensor(buf71, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf71  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf72, buf73, 4194304, grid=grid(4194304), stream=stream0)
        buf74 = reinterpret_tensor(buf72, (8192, 512), (512, 1), 0); del buf72  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (8192, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 512), (1, 512), 0), out=buf74)
        del arg49_1
        buf76 = reinterpret_tensor(buf73, (4, 2048, 512), (1048576, 512, 1), 0); del buf73  # reuse
        # Source Nodes: [add_13, forwarded_states_4, hidden_states_26, hidden_states_31, hidden_states_32, pow_6, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf54, buf59, buf74, arg5_1, buf76, 8192, 512, grid=grid(8192), stream=stream0)
        del arg5_1
        buf77 = reinterpret_tensor(buf58, (8192, 2048), (2048, 1), 0); del buf58  # reuse
        # Source Nodes: [hidden_states_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (8192, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 2048), (1, 512), 0), out=buf77)
        del arg50_1
        buf78 = reinterpret_tensor(buf77, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf77  # reuse
        # Source Nodes: [hidden_states_34], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf78, 16777216, grid=grid(16777216), stream=stream0)
        buf79 = reinterpret_tensor(buf76, (8192, 512), (512, 1), 0); del buf76  # reuse
        # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg51_1, (2048, 512), (1, 2048), 0), out=buf79)
        del arg51_1
        buf81 = reinterpret_tensor(buf64, (4, 2048, 512), (1048576, 512, 1), 0); del buf64  # reuse
        # Source Nodes: [add_15, hidden_states_26, hidden_states_31, hidden_states_39, hidden_states_40, normed_hidden_states_3, pow_7, rsqrt_6, variance_6], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf54, buf59, buf74, buf79, arg6_1, buf81, 8192, 512, grid=grid(8192), stream=stream0)
        del arg6_1
        buf82 = buf63; del buf63  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (8192, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 512), (1, 512), 0), out=buf82)
        del arg52_1
        buf83 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (8192, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 512), (1, 512), 0), out=buf83)
        del arg53_1
        buf84 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf82, buf84, 4194304, grid=grid(4194304), stream=stream0)
        buf85 = reinterpret_tensor(buf82, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf82  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf83, buf85, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf86 = reinterpret_tensor(buf69, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf69  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf85, (32, 64, 2048), (131072, 2048, 1), 0), out=buf86)
        buf89 = reinterpret_tensor(buf66, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf66  # reuse
        # Source Nodes: [softmax_3], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf86, arg36_1, buf89, 65536, 2048, grid=grid(65536), stream=stream0)
        buf90 = reinterpret_tensor(buf85, (8192, 512), (512, 1), 0); del buf85  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (8192, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf90)
        del arg54_1
        buf91 = reinterpret_tensor(buf81, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf81  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf90, buf91, 4194304, grid=grid(4194304), stream=stream0)
        buf92 = reinterpret_tensor(buf90, (32, 2048, 64), (131072, 64, 1), 0); del buf90  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf91, (32, 2048, 64), (131072, 64, 1), 0), out=buf92)
        buf93 = reinterpret_tensor(buf91, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf91  # reuse
        # Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf92, buf93, 4194304, grid=grid(4194304), stream=stream0)
        buf94 = reinterpret_tensor(buf92, (8192, 512), (512, 1), 0); del buf92  # reuse
        # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (8192, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 512), (1, 512), 0), out=buf94)
        del arg55_1
        buf95 = buf54; del buf54  # reuse
        buf97 = reinterpret_tensor(buf93, (4, 2048, 512), (1048576, 512, 1), 0); del buf93  # reuse
        # Source Nodes: [add_17, forwarded_states_6, hidden_states_26, hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_45, pow_8, rsqrt_7, variance_7], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf95, buf59, buf74, buf79, buf94, arg7_1, buf97, 8192, 512, grid=grid(8192), stream=stream0)
        del arg7_1
        buf98 = reinterpret_tensor(buf78, (8192, 2048), (2048, 1), 0); del buf78  # reuse
        # Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (8192, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 2048), (1, 512), 0), out=buf98)
        del arg56_1
        buf99 = reinterpret_tensor(buf98, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf98  # reuse
        # Source Nodes: [hidden_states_47], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf99, 16777216, grid=grid(16777216), stream=stream0)
        buf100 = reinterpret_tensor(buf97, (8192, 512), (512, 1), 0); del buf97  # reuse
        # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg57_1, (2048, 512), (1, 2048), 0), out=buf100)
        del arg57_1
        buf102 = reinterpret_tensor(buf94, (4, 2048, 512), (1048576, 512, 1), 0); del buf94  # reuse
        # Source Nodes: [add_19, hidden_states_52, hidden_states_53, normed_hidden_states_4, pow_9, rsqrt_8, variance_8], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf95, buf100, arg8_1, buf102, 8192, 512, grid=grid(8192), stream=stream0)
        del arg8_1
        buf103 = buf79; del buf79  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (8192, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 512), (1, 512), 0), out=buf103)
        del arg58_1
        buf104 = buf74; del buf74  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (8192, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 512), (1, 512), 0), out=buf104)
        del arg59_1
        buf105 = reinterpret_tensor(buf59, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf59  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf103, buf105, 4194304, grid=grid(4194304), stream=stream0)
        buf106 = reinterpret_tensor(buf103, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf103  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf104, buf106, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf107 = reinterpret_tensor(buf89, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf89  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf106, (32, 64, 2048), (131072, 2048, 1), 0), out=buf107)
        buf110 = reinterpret_tensor(buf86, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf86  # reuse
        # Source Nodes: [softmax_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf107, arg36_1, buf110, 65536, 2048, grid=grid(65536), stream=stream0)
        buf111 = reinterpret_tensor(buf106, (8192, 512), (512, 1), 0); del buf106  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (8192, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 512), (1, 512), 0), out=buf111)
        del arg60_1
        buf112 = reinterpret_tensor(buf102, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf102  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf111, buf112, 4194304, grid=grid(4194304), stream=stream0)
        buf113 = reinterpret_tensor(buf111, (32, 2048, 64), (131072, 64, 1), 0); del buf111  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf112, (32, 2048, 64), (131072, 64, 1), 0), out=buf113)
        buf114 = reinterpret_tensor(buf112, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf112  # reuse
        # Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf113, buf114, 4194304, grid=grid(4194304), stream=stream0)
        buf115 = reinterpret_tensor(buf113, (8192, 512), (512, 1), 0); del buf113  # reuse
        # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf114, (8192, 512), (512, 1), 0), reinterpret_tensor(arg61_1, (512, 512), (1, 512), 0), out=buf115)
        del arg61_1
        buf117 = reinterpret_tensor(buf114, (4, 2048, 512), (1048576, 512, 1), 0); del buf114  # reuse
        # Source Nodes: [add_21, forwarded_states_8, hidden_states_52, hidden_states_57, hidden_states_58, pow_10, rsqrt_9, variance_9], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf95, buf100, buf115, arg9_1, buf117, 8192, 512, grid=grid(8192), stream=stream0)
        del arg9_1
        buf118 = reinterpret_tensor(buf99, (8192, 2048), (2048, 1), 0); del buf99  # reuse
        # Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (8192, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 2048), (1, 512), 0), out=buf118)
        del arg62_1
        buf119 = reinterpret_tensor(buf118, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf118  # reuse
        # Source Nodes: [hidden_states_60], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf119, 16777216, grid=grid(16777216), stream=stream0)
        buf120 = reinterpret_tensor(buf117, (8192, 512), (512, 1), 0); del buf117  # reuse
        # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 512), (1, 2048), 0), out=buf120)
        del arg63_1
        buf122 = reinterpret_tensor(buf105, (4, 2048, 512), (1048576, 512, 1), 0); del buf105  # reuse
        # Source Nodes: [add_23, hidden_states_52, hidden_states_57, hidden_states_65, hidden_states_66, normed_hidden_states_5, pow_11, rsqrt_10, variance_10], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf95, buf100, buf115, buf120, arg10_1, buf122, 8192, 512, grid=grid(8192), stream=stream0)
        del arg10_1
        buf123 = buf104; del buf104  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (8192, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf123)
        del arg64_1
        buf124 = reinterpret_tensor(buf84, (8192, 512), (512, 1), 0); del buf84  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (8192, 512), (512, 1), 0), reinterpret_tensor(arg65_1, (512, 512), (1, 512), 0), out=buf124)
        del arg65_1
        buf125 = reinterpret_tensor(buf83, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf83  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf123, buf125, 4194304, grid=grid(4194304), stream=stream0)
        buf126 = reinterpret_tensor(buf123, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf123  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf124, buf126, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf127 = reinterpret_tensor(buf110, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf110  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf126, (32, 64, 2048), (131072, 2048, 1), 0), out=buf127)
        buf130 = reinterpret_tensor(buf107, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf107  # reuse
        # Source Nodes: [softmax_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf127, arg36_1, buf130, 65536, 2048, grid=grid(65536), stream=stream0)
        del arg36_1
        buf131 = reinterpret_tensor(buf126, (8192, 512), (512, 1), 0); del buf126  # reuse
        # Source Nodes: [l__mod___model_model_encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (8192, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 512), (1, 512), 0), out=buf131)
        del arg66_1
        buf132 = reinterpret_tensor(buf122, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf122  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf131, buf132, 4194304, grid=grid(4194304), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (32, 2048, 64), (131072, 64, 1), 0); del buf131  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf132, (32, 2048, 64), (131072, 64, 1), 0), out=buf133)
        buf134 = reinterpret_tensor(buf132, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf132  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf133, buf134, 4194304, grid=grid(4194304), stream=stream0)
        buf135 = reinterpret_tensor(buf133, (8192, 512), (512, 1), 0); del buf133  # reuse
        # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (8192, 512), (512, 1), 0), reinterpret_tensor(arg67_1, (512, 512), (1, 512), 0), out=buf135)
        del arg67_1
        buf136 = reinterpret_tensor(buf100, (4, 2048, 512), (1048576, 512, 1), 0); del buf100  # reuse
        buf138 = reinterpret_tensor(buf134, (4, 2048, 512), (1048576, 512, 1), 0); del buf134  # reuse
        # Source Nodes: [add_25, forwarded_states_10, hidden_states_52, hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_71, pow_12, rsqrt_11, variance_11], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf136, buf95, buf115, buf120, buf135, arg11_1, buf138, 8192, 512, grid=grid(8192), stream=stream0)
        del arg11_1
        buf139 = reinterpret_tensor(buf119, (8192, 2048), (2048, 1), 0); del buf119  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (8192, 512), (512, 1), 0), reinterpret_tensor(arg68_1, (512, 2048), (1, 512), 0), out=buf139)
        del arg68_1
        buf140 = reinterpret_tensor(buf139, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf139  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf140, 16777216, grid=grid(16777216), stream=stream0)
        buf141 = reinterpret_tensor(buf138, (8192, 512), (512, 1), 0); del buf138  # reuse
        # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg69_1, (2048, 512), (1, 2048), 0), out=buf141)
        del arg69_1
        buf143 = buf95; del buf95  # reuse
        # Source Nodes: [add_27, hidden_states_78, hidden_states_79, hidden_states_80, pow_13, rsqrt_12, variance_12], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf136, buf141, arg12_1, buf143, 8192, 512, grid=grid(8192), stream=stream0)
        del arg12_1
        buf144 = buf141; del buf141  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 512), (1, 512), 0), out=buf144)
        del arg76_1
        buf145 = reinterpret_tensor(buf136, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf136  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf18, buf145, 4194304, grid=grid(4194304), stream=stream0)
        buf146 = reinterpret_tensor(buf18, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf18  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf144, buf146, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf147 = reinterpret_tensor(buf130, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf130  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf146, (32, 64, 2048), (131072, 2048, 1), 0), out=buf147)
        buf151 = reinterpret_tensor(buf127, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf127  # reuse
        # Source Nodes: [softmax_7], Original ATen: [aten._softmax]
        triton_red_fused__softmax_16.run(buf147, buf151, 65536, 2048, grid=grid(65536), stream=stream0)
        buf150 = reinterpret_tensor(buf146, (8192, 512), (512, 1), 0); del buf146  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 512), (1, 512), 0), out=buf150)
        del arg77_1
        buf152 = buf145; del buf145  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf150, buf152, 4194304, grid=grid(4194304), stream=stream0)
        buf153 = reinterpret_tensor(buf135, (32, 2048, 64), (131072, 64, 1), 0); del buf135  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf152, (32, 2048, 64), (131072, 64, 1), 0), out=buf153)
        buf154 = reinterpret_tensor(buf152, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf152  # reuse
        # Source Nodes: [contiguous_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf153, buf154, 4194304, grid=grid(4194304), stream=stream0)
        buf155 = reinterpret_tensor(buf153, (8192, 512), (512, 1), 0); del buf153  # reuse
        # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (8192, 512), (512, 1), 0), reinterpret_tensor(arg78_1, (512, 512), (1, 512), 0), out=buf155)
        del arg78_1
        buf157 = reinterpret_tensor(buf154, (4, 2048, 512), (1048576, 512, 1), 0); del buf154  # reuse
        # Source Nodes: [add_36, forwarded_states_12, hidden_states_88, hidden_states_92, hidden_states_93, inputs_embeds_1, pow_16, rsqrt_15, variance_15], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_9.run(arg133_1, arg32_1, buf15, buf155, arg15_1, buf157, 8192, 512, grid=grid(8192), stream=stream0)
        del arg15_1
        buf158 = reinterpret_tensor(buf140, (8192, 2048), (2048, 1), 0); del buf140  # reuse
        # Source Nodes: [hidden_states_94], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (8192, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 2048), (1, 512), 0), out=buf158)
        del arg79_1
        buf159 = reinterpret_tensor(buf158, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf158  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf159, 16777216, grid=grid(16777216), stream=stream0)
        buf160 = reinterpret_tensor(buf157, (8192, 512), (512, 1), 0); del buf157  # reuse
        # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg80_1, (2048, 512), (1, 2048), 0), out=buf160)
        del arg80_1
        buf161 = reinterpret_tensor(buf15, (4, 2048, 512), (1048576, 512, 1), 0); del buf15  # reuse
        buf163 = reinterpret_tensor(buf120, (4, 2048, 512), (1048576, 512, 1), 0); del buf120  # reuse
        # Source Nodes: [add_38, hidden_states_100, hidden_states_101, hidden_states_88, hidden_states_92, inputs_embeds_1, normed_hidden_states_8, pow_17, rsqrt_16, variance_16], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10.run(buf161, arg133_1, arg32_1, buf155, buf160, arg16_1, buf163, 8192, 512, grid=grid(8192), stream=stream0)
        del arg133_1
        del arg16_1
        del arg32_1
        buf164 = buf160; del buf160  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (8192, 512), (512, 1), 0), reinterpret_tensor(arg81_1, (512, 512), (1, 512), 0), out=buf164)
        del arg81_1
        buf165 = buf155; del buf155  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (8192, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 512), (1, 512), 0), out=buf165)
        del arg82_1
        buf166 = reinterpret_tensor(buf115, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf115  # reuse
        # Source Nodes: [scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf164, buf166, 4194304, grid=grid(4194304), stream=stream0)
        buf167 = reinterpret_tensor(buf164, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf164  # reuse
        # Source Nodes: [scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf165, buf167, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf168 = reinterpret_tensor(buf151, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf151  # reuse
        # Source Nodes: [scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf167, (32, 64, 2048), (131072, 2048, 1), 0), out=buf168)
        buf170 = reinterpret_tensor(buf147, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf147  # reuse
        buf173 = reinterpret_tensor(buf6, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf6  # reuse
        # Source Nodes: [softmax_8], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf168, arg73_1, buf170, buf173, 65536, 2048, grid=grid(65536), stream=stream0)
        buf172 = reinterpret_tensor(buf167, (8192, 512), (512, 1), 0); del buf167  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (8192, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 512), (1, 512), 0), out=buf172)
        del arg83_1
        buf174 = reinterpret_tensor(buf163, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf163  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf172, buf174, 4194304, grid=grid(4194304), stream=stream0)
        buf175 = reinterpret_tensor(buf166, (32, 2048, 64), (131072, 64, 1), 0); del buf166  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf174, (32, 2048, 64), (131072, 64, 1), 0), out=buf175)
        buf176 = reinterpret_tensor(buf174, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf174  # reuse
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf175, buf176, 4194304, grid=grid(4194304), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (8192, 512), (512, 1), 0); del buf175  # reuse
        # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (8192, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 512), (1, 512), 0), out=buf177)
        del arg84_1
        buf179 = reinterpret_tensor(buf176, (4, 2048, 512), (1048576, 512, 1), 0); del buf176  # reuse
        # Source Nodes: [add_40, hidden_states_105, hidden_states_106, normed_hidden_states_9, pow_18, rsqrt_17, variance_17], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf161, buf177, arg17_1, buf179, 8192, 512, grid=grid(8192), stream=stream0)
        del arg17_1
        buf180 = reinterpret_tensor(buf125, (8192, 512), (512, 1), 0); del buf125  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (8192, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 512), (1, 512), 0), out=buf180)
        del arg85_1
        buf181 = reinterpret_tensor(buf179, (8192, 512), (512, 1), 0); del buf179  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), out=buf181)
        del arg86_1
        buf182 = reinterpret_tensor(buf124, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf180, buf182, 4194304, grid=grid(4194304), stream=stream0)
        buf183 = reinterpret_tensor(buf180, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf180  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf181, buf183, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf184 = reinterpret_tensor(buf173, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf173  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf183, (32, 64, 2048), (131072, 2048, 1), 0), out=buf184)
        buf188 = buf170; del buf170  # reuse
        # Source Nodes: [softmax_9], Original ATen: [aten._softmax]
        triton_red_fused__softmax_16.run(buf184, buf188, 65536, 2048, grid=grid(65536), stream=stream0)
        buf187 = reinterpret_tensor(buf183, (8192, 512), (512, 1), 0); del buf183  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 512), (1, 512), 0), out=buf187)
        del arg87_1
        buf189 = buf182; del buf182  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf187, buf189, 4194304, grid=grid(4194304), stream=stream0)
        buf190 = empty((32, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf188, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf189, (32, 2048, 64), (131072, 64, 1), 0), out=buf190)
        buf191 = reinterpret_tensor(buf189, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf189  # reuse
        # Source Nodes: [contiguous_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf190, buf191, 4194304, grid=grid(4194304), stream=stream0)
        buf192 = reinterpret_tensor(buf190, (8192, 512), (512, 1), 0); del buf190  # reuse
        # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (8192, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 512), (1, 512), 0), out=buf192)
        del arg88_1
        buf194 = reinterpret_tensor(buf191, (4, 2048, 512), (1048576, 512, 1), 0); del buf191  # reuse
        # Source Nodes: [add_42, forwarded_states_14, hidden_states_105, hidden_states_109, hidden_states_110, pow_19, rsqrt_18, variance_18], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf161, buf177, buf192, arg18_1, buf194, 8192, 512, grid=grid(8192), stream=stream0)
        del arg18_1
        buf195 = reinterpret_tensor(buf159, (8192, 2048), (2048, 1), 0); del buf159  # reuse
        # Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (8192, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 2048), (1, 512), 0), out=buf195)
        del arg89_1
        buf196 = reinterpret_tensor(buf195, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf195  # reuse
        # Source Nodes: [hidden_states_112], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf196, 16777216, grid=grid(16777216), stream=stream0)
        buf197 = reinterpret_tensor(buf194, (8192, 512), (512, 1), 0); del buf194  # reuse
        # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg90_1, (2048, 512), (1, 2048), 0), out=buf197)
        del arg90_1
        buf199 = empty((4, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, hidden_states_105, hidden_states_109, hidden_states_117, hidden_states_118, normed_hidden_states_10, pow_20, rsqrt_19, variance_19], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf161, buf177, buf192, buf197, arg19_1, buf199, 8192, 512, grid=grid(8192), stream=stream0)
        del arg19_1
        buf200 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (8192, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 512), (1, 512), 0), out=buf200)
        del arg91_1
        buf201 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (8192, 512), (512, 1), 0), reinterpret_tensor(arg92_1, (512, 512), (1, 512), 0), out=buf201)
        del arg92_1
        buf202 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf200, buf202, 4194304, grid=grid(4194304), stream=stream0)
        buf203 = reinterpret_tensor(buf200, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf200  # reuse
        # Source Nodes: [scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf201, buf203, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf204 = reinterpret_tensor(buf188, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf188  # reuse
        # Source Nodes: [scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf202, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf203, (32, 64, 2048), (131072, 2048, 1), 0), out=buf204)
        buf206 = reinterpret_tensor(buf184, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf184  # reuse
        buf209 = reinterpret_tensor(buf168, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf168  # reuse
        # Source Nodes: [softmax_10], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf204, arg73_1, buf206, buf209, 65536, 2048, grid=grid(65536), stream=stream0)
        buf208 = reinterpret_tensor(buf203, (8192, 512), (512, 1), 0); del buf203  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (8192, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 512), (1, 512), 0), out=buf208)
        del arg93_1
        buf210 = reinterpret_tensor(buf199, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf199  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf208, buf210, 4194304, grid=grid(4194304), stream=stream0)
        buf211 = reinterpret_tensor(buf202, (32, 2048, 64), (131072, 64, 1), 0); del buf202  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf210, (32, 2048, 64), (131072, 64, 1), 0), out=buf211)
        buf212 = reinterpret_tensor(buf210, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf210  # reuse
        # Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf211, buf212, 4194304, grid=grid(4194304), stream=stream0)
        buf213 = reinterpret_tensor(buf211, (8192, 512), (512, 1), 0); del buf211  # reuse
        # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (8192, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 512), (1, 512), 0), out=buf213)
        del arg94_1
        buf214 = buf161; del buf161  # reuse
        buf216 = reinterpret_tensor(buf212, (4, 2048, 512), (1048576, 512, 1), 0); del buf212  # reuse
        # Source Nodes: [add_46, hidden_states_105, hidden_states_109, hidden_states_117, hidden_states_122, hidden_states_123, normed_hidden_states_11, pow_21, rsqrt_20, variance_20], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf214, buf177, buf192, buf197, buf213, arg20_1, buf216, 8192, 512, grid=grid(8192), stream=stream0)
        del arg20_1
        buf217 = buf213; del buf213  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (8192, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 512), (1, 512), 0), out=buf217)
        del arg95_1
        buf218 = reinterpret_tensor(buf216, (8192, 512), (512, 1), 0); del buf216  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 512), (1, 512), 0), out=buf218)
        del arg96_1
        buf219 = reinterpret_tensor(buf197, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf217, buf219, 4194304, grid=grid(4194304), stream=stream0)
        buf220 = reinterpret_tensor(buf217, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf217  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf218, buf220, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf221 = reinterpret_tensor(buf209, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf209  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf219, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf220, (32, 64, 2048), (131072, 2048, 1), 0), out=buf221)
        buf225 = buf206; del buf206  # reuse
        # Source Nodes: [softmax_11], Original ATen: [aten._softmax]
        triton_red_fused__softmax_16.run(buf221, buf225, 65536, 2048, grid=grid(65536), stream=stream0)
        buf224 = reinterpret_tensor(buf220, (8192, 512), (512, 1), 0); del buf220  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg97_1, (512, 512), (1, 512), 0), out=buf224)
        del arg97_1
        buf226 = buf219; del buf219  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf224, buf226, 4194304, grid=grid(4194304), stream=stream0)
        buf227 = reinterpret_tensor(buf192, (32, 2048, 64), (131072, 64, 1), 0); del buf192  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf226, (32, 2048, 64), (131072, 64, 1), 0), out=buf227)
        buf228 = reinterpret_tensor(buf226, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf226  # reuse
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf227, buf228, 4194304, grid=grid(4194304), stream=stream0)
        buf229 = reinterpret_tensor(buf227, (8192, 512), (512, 1), 0); del buf227  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (8192, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 512), (1, 512), 0), out=buf229)
        del arg98_1
        buf231 = reinterpret_tensor(buf228, (4, 2048, 512), (1048576, 512, 1), 0); del buf228  # reuse
        # Source Nodes: [add_48, forwarded_states_16, hidden_states_126, hidden_states_127, pow_22, rsqrt_21, variance_21], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf214, buf229, arg21_1, buf231, 8192, 512, grid=grid(8192), stream=stream0)
        del arg21_1
        buf232 = reinterpret_tensor(buf196, (8192, 2048), (2048, 1), 0); del buf196  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (8192, 512), (512, 1), 0), reinterpret_tensor(arg99_1, (512, 2048), (1, 512), 0), out=buf232)
        del arg99_1
        buf233 = reinterpret_tensor(buf232, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf232  # reuse
        # Source Nodes: [hidden_states_129], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf233, 16777216, grid=grid(16777216), stream=stream0)
        buf234 = reinterpret_tensor(buf231, (8192, 512), (512, 1), 0); del buf231  # reuse
        # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg100_1, (2048, 512), (1, 2048), 0), out=buf234)
        del arg100_1
        buf236 = reinterpret_tensor(buf177, (4, 2048, 512), (1048576, 512, 1), 0); del buf177  # reuse
        # Source Nodes: [add_50, hidden_states_126, hidden_states_134, hidden_states_135, normed_hidden_states_12, pow_23, rsqrt_22, variance_22], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf214, buf229, buf234, arg22_1, buf236, 8192, 512, grid=grid(8192), stream=stream0)
        del arg22_1
        buf237 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (8192, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 512), (1, 512), 0), out=buf237)
        del arg101_1
        buf238 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (8192, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 512), (1, 512), 0), out=buf238)
        del arg102_1
        buf239 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf237, buf239, 4194304, grid=grid(4194304), stream=stream0)
        buf240 = reinterpret_tensor(buf237, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf237  # reuse
        # Source Nodes: [scores_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf238, buf240, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf241 = reinterpret_tensor(buf225, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf225  # reuse
        # Source Nodes: [scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf240, (32, 64, 2048), (131072, 2048, 1), 0), out=buf241)
        buf243 = reinterpret_tensor(buf221, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf221  # reuse
        buf246 = reinterpret_tensor(buf204, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf204  # reuse
        # Source Nodes: [softmax_12], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf241, arg73_1, buf243, buf246, 65536, 2048, grid=grid(65536), stream=stream0)
        buf245 = reinterpret_tensor(buf240, (8192, 512), (512, 1), 0); del buf240  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (8192, 512), (512, 1), 0), reinterpret_tensor(arg103_1, (512, 512), (1, 512), 0), out=buf245)
        del arg103_1
        buf247 = reinterpret_tensor(buf236, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf245, buf247, 4194304, grid=grid(4194304), stream=stream0)
        buf248 = reinterpret_tensor(buf239, (32, 2048, 64), (131072, 64, 1), 0); del buf239  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf247, (32, 2048, 64), (131072, 64, 1), 0), out=buf248)
        buf249 = reinterpret_tensor(buf247, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf247  # reuse
        # Source Nodes: [contiguous_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf248, buf249, 4194304, grid=grid(4194304), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (8192, 512), (512, 1), 0); del buf248  # reuse
        # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (8192, 512), (512, 1), 0), reinterpret_tensor(arg104_1, (512, 512), (1, 512), 0), out=buf250)
        del arg104_1
        buf252 = reinterpret_tensor(buf249, (4, 2048, 512), (1048576, 512, 1), 0); del buf249  # reuse
        # Source Nodes: [add_52, hidden_states_126, hidden_states_134, hidden_states_139, hidden_states_140, normed_hidden_states_13, pow_24, rsqrt_23, variance_23], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf214, buf229, buf234, buf250, arg23_1, buf252, 8192, 512, grid=grid(8192), stream=stream0)
        del arg23_1
        buf253 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (8192, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 512), (1, 512), 0), out=buf253)
        del arg105_1
        buf254 = reinterpret_tensor(buf252, (8192, 512), (512, 1), 0); del buf252  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 512), (1, 512), 0), out=buf254)
        del arg106_1
        buf255 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf253, buf255, 4194304, grid=grid(4194304), stream=stream0)
        buf256 = reinterpret_tensor(buf253, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf253  # reuse
        # Source Nodes: [scores_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf254, buf256, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf257 = reinterpret_tensor(buf246, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf246  # reuse
        # Source Nodes: [scores_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf255, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf256, (32, 64, 2048), (131072, 2048, 1), 0), out=buf257)
        buf261 = buf243; del buf243  # reuse
        # Source Nodes: [softmax_13], Original ATen: [aten._softmax]
        triton_red_fused__softmax_16.run(buf257, buf261, 65536, 2048, grid=grid(65536), stream=stream0)
        buf260 = reinterpret_tensor(buf256, (8192, 512), (512, 1), 0); del buf256  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 512), (1, 512), 0), out=buf260)
        del arg107_1
        buf262 = buf255; del buf255  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf260, buf262, 4194304, grid=grid(4194304), stream=stream0)
        buf263 = empty((32, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf262, (32, 2048, 64), (131072, 64, 1), 0), out=buf263)
        buf264 = reinterpret_tensor(buf262, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf262  # reuse
        # Source Nodes: [contiguous_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf263, buf264, 4194304, grid=grid(4194304), stream=stream0)
        buf265 = reinterpret_tensor(buf263, (8192, 512), (512, 1), 0); del buf263  # reuse
        # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (8192, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 512), (1, 512), 0), out=buf265)
        del arg108_1
        buf266 = buf214; del buf214  # reuse
        buf268 = reinterpret_tensor(buf264, (4, 2048, 512), (1048576, 512, 1), 0); del buf264  # reuse
        # Source Nodes: [add_54, forwarded_states_18, hidden_states_126, hidden_states_134, hidden_states_139, hidden_states_143, hidden_states_144, pow_25, rsqrt_24, variance_24], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf266, buf229, buf234, buf250, buf265, arg24_1, buf268, 8192, 512, grid=grid(8192), stream=stream0)
        del arg24_1
        buf269 = reinterpret_tensor(buf233, (8192, 2048), (2048, 1), 0); del buf233  # reuse
        # Source Nodes: [hidden_states_145], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (8192, 512), (512, 1), 0), reinterpret_tensor(arg109_1, (512, 2048), (1, 512), 0), out=buf269)
        del arg109_1
        buf270 = reinterpret_tensor(buf269, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf269  # reuse
        # Source Nodes: [hidden_states_146], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf270, 16777216, grid=grid(16777216), stream=stream0)
        buf271 = reinterpret_tensor(buf268, (8192, 512), (512, 1), 0); del buf268  # reuse
        # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 512), (1, 2048), 0), out=buf271)
        del arg110_1
        buf273 = reinterpret_tensor(buf265, (4, 2048, 512), (1048576, 512, 1), 0); del buf265  # reuse
        # Source Nodes: [add_56, hidden_states_151, hidden_states_152, normed_hidden_states_14, pow_26, rsqrt_25, variance_25], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf266, buf271, arg25_1, buf273, 8192, 512, grid=grid(8192), stream=stream0)
        del arg25_1
        buf274 = buf250; del buf250  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (8192, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 512), (1, 512), 0), out=buf274)
        del arg111_1
        buf275 = buf234; del buf234  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (8192, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 512), (1, 512), 0), out=buf275)
        del arg112_1
        buf276 = reinterpret_tensor(buf229, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf229  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf274, buf276, 4194304, grid=grid(4194304), stream=stream0)
        buf277 = reinterpret_tensor(buf274, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf274  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf275, buf277, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf278 = reinterpret_tensor(buf261, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf261  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf276, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf277, (32, 64, 2048), (131072, 2048, 1), 0), out=buf278)
        buf280 = reinterpret_tensor(buf257, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf257  # reuse
        buf283 = reinterpret_tensor(buf241, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf241  # reuse
        # Source Nodes: [softmax_14], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf278, arg73_1, buf280, buf283, 65536, 2048, grid=grid(65536), stream=stream0)
        buf282 = reinterpret_tensor(buf277, (8192, 512), (512, 1), 0); del buf277  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (8192, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 512), (1, 512), 0), out=buf282)
        del arg113_1
        buf284 = reinterpret_tensor(buf273, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf273  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf282, buf284, 4194304, grid=grid(4194304), stream=stream0)
        buf285 = reinterpret_tensor(buf276, (32, 2048, 64), (131072, 64, 1), 0); del buf276  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf283, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf284, (32, 2048, 64), (131072, 64, 1), 0), out=buf285)
        buf286 = reinterpret_tensor(buf284, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf284  # reuse
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf285, buf286, 4194304, grid=grid(4194304), stream=stream0)
        buf287 = reinterpret_tensor(buf285, (8192, 512), (512, 1), 0); del buf285  # reuse
        # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (8192, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 512), (1, 512), 0), out=buf287)
        del arg114_1
        buf289 = reinterpret_tensor(buf286, (4, 2048, 512), (1048576, 512, 1), 0); del buf286  # reuse
        # Source Nodes: [add_58, hidden_states_151, hidden_states_156, hidden_states_157, normed_hidden_states_15, pow_27, rsqrt_26, variance_26], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf266, buf271, buf287, arg26_1, buf289, 8192, 512, grid=grid(8192), stream=stream0)
        del arg26_1
        buf290 = empty((8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___model_model_decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (8192, 512), (512, 1), 0), reinterpret_tensor(arg115_1, (512, 512), (1, 512), 0), out=buf290)
        del arg115_1
        buf291 = reinterpret_tensor(buf289, (8192, 512), (512, 1), 0); del buf289  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 512), (1, 512), 0), out=buf291)
        del arg116_1
        buf292 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf290, buf292, 4194304, grid=grid(4194304), stream=stream0)
        buf293 = reinterpret_tensor(buf290, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf290  # reuse
        # Source Nodes: [scores_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf291, buf293, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf294 = reinterpret_tensor(buf283, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf283  # reuse
        # Source Nodes: [scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf293, (32, 64, 2048), (131072, 2048, 1), 0), out=buf294)
        buf298 = buf280; del buf280  # reuse
        # Source Nodes: [softmax_15], Original ATen: [aten._softmax]
        triton_red_fused__softmax_16.run(buf294, buf298, 65536, 2048, grid=grid(65536), stream=stream0)
        buf297 = reinterpret_tensor(buf293, (8192, 512), (512, 1), 0); del buf293  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 512), (1, 512), 0), out=buf297)
        del arg117_1
        buf299 = buf292; del buf292  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf297, buf299, 4194304, grid=grid(4194304), stream=stream0)
        buf300 = empty((32, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf298, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf299, (32, 2048, 64), (131072, 64, 1), 0), out=buf300)
        buf301 = reinterpret_tensor(buf299, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf299  # reuse
        # Source Nodes: [contiguous_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf300, buf301, 4194304, grid=grid(4194304), stream=stream0)
        buf302 = reinterpret_tensor(buf300, (8192, 512), (512, 1), 0); del buf300  # reuse
        # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (8192, 512), (512, 1), 0), reinterpret_tensor(arg118_1, (512, 512), (1, 512), 0), out=buf302)
        del arg118_1
        buf304 = reinterpret_tensor(buf301, (4, 2048, 512), (1048576, 512, 1), 0); del buf301  # reuse
        # Source Nodes: [add_60, forwarded_states_20, hidden_states_151, hidden_states_156, hidden_states_160, hidden_states_161, pow_28, rsqrt_27, variance_27], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf266, buf271, buf287, buf302, arg27_1, buf304, 8192, 512, grid=grid(8192), stream=stream0)
        del arg27_1
        buf305 = reinterpret_tensor(buf270, (8192, 2048), (2048, 1), 0); del buf270  # reuse
        # Source Nodes: [hidden_states_162], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (8192, 512), (512, 1), 0), reinterpret_tensor(arg119_1, (512, 2048), (1, 512), 0), out=buf305)
        del arg119_1
        buf306 = reinterpret_tensor(buf305, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf305  # reuse
        # Source Nodes: [hidden_states_163], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf306, 16777216, grid=grid(16777216), stream=stream0)
        buf307 = reinterpret_tensor(buf304, (8192, 512), (512, 1), 0); del buf304  # reuse
        # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf306, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg120_1, (2048, 512), (1, 2048), 0), out=buf307)
        del arg120_1
        buf308 = buf266; del buf266  # reuse
        buf310 = empty((4, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62, hidden_states_151, hidden_states_156, hidden_states_160, hidden_states_168, hidden_states_169, normed_hidden_states_16, pow_29, rsqrt_28, variance_28], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf308, buf271, buf287, buf302, buf307, arg28_1, buf310, 8192, 512, grid=grid(8192), stream=stream0)
        del arg28_1
        buf311 = buf307; del buf307  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (8192, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 512), (1, 512), 0), out=buf311)
        del arg121_1
        buf312 = buf302; del buf302  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (8192, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 512), (1, 512), 0), out=buf312)
        del arg122_1
        buf313 = reinterpret_tensor(buf287, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf287  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf311, buf313, 4194304, grid=grid(4194304), stream=stream0)
        buf314 = reinterpret_tensor(buf311, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf311  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf312, buf314, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf315 = reinterpret_tensor(buf298, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf298  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf313, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf314, (32, 64, 2048), (131072, 2048, 1), 0), out=buf315)
        buf317 = reinterpret_tensor(buf294, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf294  # reuse
        buf320 = reinterpret_tensor(buf278, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), 0); del buf278  # reuse
        # Source Nodes: [softmax_16], Original ATen: [aten._softmax]
        triton_red_fused__softmax_3.run(buf315, arg73_1, buf317, buf320, 65536, 2048, grid=grid(65536), stream=stream0)
        del arg73_1
        del buf315
        buf319 = reinterpret_tensor(buf314, (8192, 512), (512, 1), 0); del buf314  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (8192, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 512), (1, 512), 0), out=buf319)
        del arg123_1
        buf321 = reinterpret_tensor(buf310, (4, 8, 2048, 64), (1048576, 131072, 64, 1), 0); del buf310  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf319, buf321, 4194304, grid=grid(4194304), stream=stream0)
        buf322 = reinterpret_tensor(buf313, (32, 2048, 64), (131072, 64, 1), 0); del buf313  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf321, (32, 2048, 64), (131072, 64, 1), 0), out=buf322)
        buf323 = reinterpret_tensor(buf321, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf321  # reuse
        # Source Nodes: [contiguous_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf322, buf323, 4194304, grid=grid(4194304), stream=stream0)
        buf324 = reinterpret_tensor(buf322, (8192, 512), (512, 1), 0); del buf322  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (8192, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 512), (1, 512), 0), out=buf324)
        del arg124_1
        buf326 = reinterpret_tensor(buf323, (4, 2048, 512), (1048576, 512, 1), 0); del buf323  # reuse
        # Source Nodes: [add_64, hidden_states_173, hidden_states_174, normed_hidden_states_17, pow_30, rsqrt_29, variance_29], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf308, buf324, arg29_1, buf326, 8192, 512, grid=grid(8192), stream=stream0)
        del arg29_1
        buf327 = buf271; del buf271  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (8192, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 512), (1, 512), 0), out=buf327)
        del arg125_1
        buf328 = reinterpret_tensor(buf326, (8192, 512), (512, 1), 0); del buf326  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 512), (1, 512), 0), out=buf328)
        del arg126_1
        buf329 = empty((4, 8, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf327, buf329, 4194304, grid=grid(4194304), stream=stream0)
        buf330 = reinterpret_tensor(buf327, (4, 8, 64, 2048), (1048576, 131072, 2048, 1), 0); del buf327  # reuse
        # Source Nodes: [scores_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf328, buf330, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf331 = reinterpret_tensor(buf320, (32, 2048, 2048), (4194304, 2048, 1), 0); del buf320  # reuse
        # Source Nodes: [scores_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf329, (32, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf330, (32, 64, 2048), (131072, 2048, 1), 0), out=buf331)
        buf335 = buf317; del buf317  # reuse
        # Source Nodes: [softmax_17], Original ATen: [aten._softmax]
        triton_red_fused__softmax_16.run(buf331, buf335, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf331
        buf334 = reinterpret_tensor(buf330, (8192, 512), (512, 1), 0); del buf330  # reuse
        # Source Nodes: [l__mod___model_model_decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (8192, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 512), (1, 512), 0), out=buf334)
        del arg127_1
        buf336 = buf329; del buf329  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf334, buf336, 4194304, grid=grid(4194304), stream=stream0)
        buf337 = empty((32, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (32, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf336, (32, 2048, 64), (131072, 64, 1), 0), out=buf337)
        del buf335
        buf338 = reinterpret_tensor(buf336, (4, 2048, 8, 64), (1048576, 512, 64, 1), 0); del buf336  # reuse
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf337, buf338, 4194304, grid=grid(4194304), stream=stream0)
        buf339 = reinterpret_tensor(buf337, (8192, 512), (512, 1), 0); del buf337  # reuse
        # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (8192, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 512), (1, 512), 0), out=buf339)
        del arg128_1
        buf341 = reinterpret_tensor(buf338, (4, 2048, 512), (1048576, 512, 1), 0); del buf338  # reuse
        # Source Nodes: [add_66, forwarded_states_22, hidden_states_173, hidden_states_177, hidden_states_178, pow_31, rsqrt_30, variance_30], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf308, buf324, buf339, arg30_1, buf341, 8192, 512, grid=grid(8192), stream=stream0)
        del arg30_1
        buf342 = reinterpret_tensor(buf306, (8192, 2048), (2048, 1), 0); del buf306  # reuse
        # Source Nodes: [hidden_states_179], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (8192, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 2048), (1, 512), 0), out=buf342)
        del arg129_1
        buf343 = reinterpret_tensor(buf342, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf342  # reuse
        # Source Nodes: [hidden_states_180], Original ATen: [aten.relu]
        triton_poi_fused_relu_8.run(buf343, 16777216, grid=grid(16777216), stream=stream0)
        buf344 = reinterpret_tensor(buf341, (8192, 512), (512, 1), 0); del buf341  # reuse
        # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (8192, 2048), (2048, 1), 0), reinterpret_tensor(arg130_1, (2048, 512), (1, 2048), 0), out=buf344)
        del arg130_1
        del buf343
        buf346 = empty((4, 2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68, hidden_states_173, hidden_states_177, hidden_states_185, hidden_states_186, hidden_states_187, pow_32, rsqrt_31, sequence_output_1, variance_31], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_17.run(buf308, buf324, buf339, buf344, arg31_1, buf346, 8192, 512, grid=grid(8192), stream=stream0)
        del arg31_1
        del buf308
        del buf324
        del buf339
        del buf344
        buf347 = empty((8192, 32128), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (8192, 512), (512, 1), 0), reinterpret_tensor(arg131_1, (512, 32128), (1, 512), 0), out=buf347)
        del arg131_1
        return (reinterpret_tensor(buf347, (4, 2048, 32128), (65798144, 32128, 1), 0), reinterpret_tensor(buf3, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf10, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf144, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf150, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf165, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf172, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf181, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf187, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf201, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf208, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf218, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf224, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf238, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf245, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf254, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf260, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf275, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf282, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf291, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf297, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf312, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf319, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf328, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), reinterpret_tensor(buf334, (4, 8, 2048, 64), (1048576, 64, 512, 1), 0), buf143, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    arg133_1 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_T5', benchmark_compiled_module)
