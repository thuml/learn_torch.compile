
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


# kernel path: /tmp/torchinductor_youkaichao/3b/c3b7ybvbnfnrtwcj7pkfmgkhpcdzwsifvmvphbqxpvyf3ynimg7d.py
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 32128
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp3 < 32128")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp0 + 32128
        tmp11 = tmp0 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp0)
        tl.device_assert(((0 <= tmp12) & (tmp12 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp12 < 32128")
        tmp13 = tl.load(in_ptr1 + (r1 + (512*tmp12)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = 512.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-06
        tmp17 = tmp15 + tmp16
        tmp18 = tl.math.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp9 * tmp19
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp20, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ph/cphluw3wyobfnibqsyb52abdpw3le2kxtqvu4mqsqp6vq5d533p3.py
# Source Nodes: [softmax_6], Original ATen: [aten._softmax]
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
triton_red_fused__softmax_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp33 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = (-1)*(tl.math.min(0, r2 + ((-1)*x0)))
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
        tmp22 = r2
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
        r2 = rindex
        tmp35 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = (-1)*(tl.math.min(0, r2 + ((-1)*x0)))
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
        tmp57 = r2
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
        tl.store(out_ptr1 + (r2 + (1024*x3)), tmp67, rmask)
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp72 = tl.load(out_ptr1 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.exp(tmp72)
        tmp74 = tmp73 / tmp70
        tl.store(out_ptr3 + (r2 + (1024*x3)), tmp74, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv257spctbovirf6lrmpvlj4kb2buppp5kiqmi6xbpt2mjpddh32.py
# Source Nodes: [contiguous_6], Original ATen: [aten.clone]
# contiguous_6 => clone_34
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
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civkqmjrakvigzldwb2w46zdtqj6itffmd3iecxb3c6jl5ot6qif.py
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
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 32128
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp3 < 32128")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp11 + 32128
        tmp13 = tmp11 < 0
        tmp14 = tl.where(tmp13, tmp12, tmp11)
        tl.device_assert(((0 <= tmp14) & (tmp14 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp14 < 32128")
        tmp15 = tl.load(in_ptr1 + (r1 + (512*tmp14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp0 + 32128
        tmp22 = tmp0 < 0
        tmp23 = tl.where(tmp22, tmp21, tmp0)
        tl.device_assert(((0 <= tmp23) & (tmp23 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp23 < 32128")
        tmp24 = tl.load(in_ptr1 + (r1 + (512*tmp23)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.device_assert(((0 <= tmp37) & (tmp37 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp37 < 32128")
        tmp38 = tl.load(in_ptr1 + (r1 + (512*tmp37)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tmp18 / tmp27
        tmp40 = tmp39 + tmp29
        tmp41 = tl.math.rsqrt(tmp40)
        tmp42 = tmp38 * tmp41
        tmp43 = tmp34 * tmp42
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp33, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp43, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czd6baok6jrcb6ihgipfaicnhwetrjkydc7qc4mdniuo44hvqasb.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, div_2, exp, sub_2, sum_1
triton_red_fused__softmax_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp30 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r2 + ((-1)*x0)
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
        tmp28 = tmp0 + tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = triton_helpers.maximum(_tmp30, tmp29)
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp30 = triton_helpers.max2(_tmp30, 1)[:, None]
    _tmp64 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp32 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = r2 + ((-1)*x0)
        tmp34 = tl.full([1, 1], 0, tl.int64)
        tmp35 = tmp33 > tmp34
        tmp36 = tmp35.to(tl.int64)
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 * tmp37
        tmp39 = tmp38 + tmp34
        tmp40 = tl.abs(tmp33)
        tmp41 = tl.full([1, 1], 8, tl.int64)
        tmp42 = tmp40 < tmp41
        tmp43 = tmp40.to(tl.float32)
        tmp44 = 8.0
        tmp45 = tmp43 / tmp44
        tmp46 = tl.log(tmp45)
        tmp47 = 2.772588722239781
        tmp48 = tmp46 / tmp47
        tmp49 = tmp48 * tmp44
        tmp50 = tmp49.to(tl.int64)
        tmp51 = tmp50 + tmp41
        tmp52 = tl.full([1, 1], 15, tl.int64)
        tmp53 = triton_helpers.minimum(tmp51, tmp52)
        tmp54 = tl.where(tmp42, tmp40, tmp53)
        tmp55 = tmp39 + tmp54
        tmp56 = tmp55 + 32
        tmp57 = tmp55 < 0
        tmp58 = tl.where(tmp57, tmp56, tmp55)
        tl.device_assert(((0 <= tmp58) & (tmp58 < 32)) | ~rmask, "index out of bounds: 0 <= tmp58 < 32")
        tmp59 = tl.load(in_ptr1 + (x1 + (8*tmp58)), rmask, eviction_policy='evict_last', other=0.0)
        tmp60 = tmp32 + tmp59
        tmp61 = tmp60 - tmp30
        tmp62 = tl.exp(tmp61)
        tmp63 = tl.broadcast_to(tmp62, [XBLOCK, RBLOCK])
        tmp65 = _tmp64 + tmp63
        _tmp64 = tl.where(rmask, tmp65, _tmp64)
    tmp64 = tl.sum(_tmp64, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp66 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp67 = r2 + ((-1)*x0)
        tmp68 = tl.full([1, 1], 0, tl.int64)
        tmp69 = tmp67 > tmp68
        tmp70 = tmp69.to(tl.int64)
        tmp71 = tl.full([1, 1], 16, tl.int64)
        tmp72 = tmp70 * tmp71
        tmp73 = tmp72 + tmp68
        tmp74 = tl.abs(tmp67)
        tmp75 = tl.full([1, 1], 8, tl.int64)
        tmp76 = tmp74 < tmp75
        tmp77 = tmp74.to(tl.float32)
        tmp78 = 8.0
        tmp79 = tmp77 / tmp78
        tmp80 = tl.log(tmp79)
        tmp81 = 2.772588722239781
        tmp82 = tmp80 / tmp81
        tmp83 = tmp82 * tmp78
        tmp84 = tmp83.to(tl.int64)
        tmp85 = tmp84 + tmp75
        tmp86 = tl.full([1, 1], 15, tl.int64)
        tmp87 = triton_helpers.minimum(tmp85, tmp86)
        tmp88 = tl.where(tmp76, tmp74, tmp87)
        tmp89 = tmp73 + tmp88
        tmp90 = tmp89 + 32
        tmp91 = tmp89 < 0
        tmp92 = tl.where(tmp91, tmp90, tmp89)
        tl.device_assert(((0 <= tmp92) & (tmp92 < 32)) | ~rmask, "index out of bounds: 0 <= tmp92 < 32")
        tmp93 = tl.load(in_ptr1 + (x1 + (8*tmp92)), rmask, eviction_policy='evict_last', other=0.0)
        tmp94 = tmp66 + tmp93
        tmp95 = tmp94 - tmp30
        tmp96 = tl.exp(tmp95)
        tmp97 = tmp96 / tmp64
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp97, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chg3heoe3nnofo4t3onqsfx6j4vwgc6vtrbsqwzzueose6v35yzx.py
# Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, inputs_embeds, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_5 => add_7
# forwarded_states => mul_6
# hidden_states_5 => add_6
# hidden_states_6 => mul_5
# inputs_embeds => embedding
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# variance_1 => mean_1
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 32128
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp3 < 32128")
    tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.rsqrt(tmp16)
    tmp18 = tmp6 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/cccftrpjnmbdgitse3i24k6oml447kuj4ochz2sktztxrayxfeqi.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfp5pprwi3zcp5vr7gd76jn6ejhzn2odbttycbd3pamhqgz6lkap.py
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
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 32128
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp3 < 32128")
    tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/czi6kadjlvjsmsuloriiv366q5k3whhic4i7xmu6epcpmr2btpez.py
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
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 32128
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp3 < 32128")
    tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 512.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp10 * tmp21
    tmp23 = tmp16 * tmp22
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ui/cuibj3lxo7oxlqpeorhg5smnfbgejefdoatlvzxhvv2qp3jkvxue.py
# Source Nodes: [add_11, hidden_states_26, hidden_states_27, normed_hidden_states_2, pow_5, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_11 => add_14
# hidden_states_26 => add_13
# hidden_states_27 => mul_11
# normed_hidden_states_2 => mul_12
# pow_5 => pow_5
# rsqrt_4 => rsqrt_4
# variance_4 => mean_4
triton_per_fused_add_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = 512.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = tmp2 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp15, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzlybu6qbyve2s22kjnz4eyjrrd4fedqo7qoq7qjlodyivm4zah.py
# Source Nodes: [add_13, forwarded_states_4, hidden_states_26, hidden_states_31, hidden_states_32, pow_6, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_13 => add_17
# forwarded_states_4 => mul_14
# hidden_states_26 => add_13
# hidden_states_31 => add_16
# hidden_states_32 => mul_13
# pow_6 => pow_6
# rsqrt_5 => rsqrt_5
# variance_5 => mean_5
triton_per_fused_add_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 512.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp16 = tmp4 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxnx5dyiuurtyz3lfctec7cwnu3wguijwutsw3ps2yjo6prer3a.py
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
triton_per_fused_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.rsqrt(tmp16)
    tmp18 = tmp6 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c62owvxhv36ppzxkwhcp2mnc3pte6hjimifjw57qoallr4emqipt.py
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
triton_per_fused_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czgna7v3sl5vyg7df3ciwaiugmcx6yonocaox43jcooo7g4nd5qq.py
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
triton_per_fused_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6nkfpbswpefualvq5wuxta7mhqroerrawjzypg2reymh3ffkdj.py
# Source Nodes: [softmax_7], Original ATen: [aten._softmax]
# softmax_7 => amax_7, div_11, exp_7, sub_12, sum_8
triton_per_fused__softmax_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp3, 0))
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp11, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oh/cohvlmd7dsskog42seq3h26f5olljf6m4eudworzj35sgvbusczg.py
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
triton_per_fused_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
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
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c42pnerfoaeylmjk5zgd6z5ovksu7uzzwohcfjgfjzomjkhxzlpq.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_18, exp_18, sub_23, sum_19
triton_red_fused__log_softmax_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 32128
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (32128*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckmpxpwm6xhqo2fbgtet5v6yzxvtoktariallordfaspk53cvhhw.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_7, div_22, full_default_7, ne_1, ne_2, neg_1, sum_20, sum_21, where_3
triton_per_fused_nll_loss_forward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 32128
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 32128), "index out of bounds: 0 <= tmp7 < 32128")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (32128*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1 = args
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
    assert_size_stride(arg132_1, (1, 1024), (1024, 1))
    assert_size_stride(arg133_1, (1, 1024), (1024, 1))
    assert_size_stride(arg134_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_28, hidden_states_84, inputs_embeds_1, normed_hidden_states_6, pow_14, rsqrt_13, variance_13], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0.run(arg134_1, arg32_1, arg13_1, buf1, 1024, 512, grid=grid(1024), stream=stream0)
        del arg13_1
        buf2 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1024, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 512), (1, 512), 0), out=buf2)
        del arg70_1
        buf3 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1024, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 512), (1, 512), 0), out=buf3)
        del arg71_1
        buf4 = empty((8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf3, (8, 64, 1024), (64, 1, 512), 0), out=buf4)
        buf6 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        buf9 = empty((1, 8, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_6], Original ATen: [aten._softmax]
        triton_red_fused__softmax_1.run(buf4, arg73_1, buf6, buf9, 8192, 1024, grid=grid(8192), stream=stream0)
        buf8 = buf2; del buf2  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1024, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), out=buf8)
        del arg72_1
        buf10 = reinterpret_tensor(buf1, (8, 1024, 64), (65536, 64, 1), 0); del buf1  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf8, (8, 1024, 64), (64, 512, 1), 0), out=buf10)
        buf11 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf10, buf11, 524288, grid=grid(524288), stream=stream0)
        buf12 = reinterpret_tensor(buf10, (1024, 512), (512, 1), 0); del buf10  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (1024, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 512), (1, 512), 0), out=buf12)
        del arg74_1
        buf14 = reinterpret_tensor(buf11, (1, 1024, 512), (524288, 512, 1), 0); del buf11  # reuse
        buf17 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_33, hidden_states_1, hidden_states_88, hidden_states_89, inputs_embeds, inputs_embeds_1, normed_hidden_states, normed_hidden_states_7, pow_1, pow_15, rsqrt, rsqrt_14, variance, variance_14], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_3.run(arg134_1, arg32_1, buf12, arg132_1, arg14_1, arg0_1, buf14, buf17, 1024, 512, grid=grid(1024), stream=stream0)
        del arg0_1
        del arg14_1
        buf15 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (1024, 512), (512, 1), 0), reinterpret_tensor(arg75_1, (512, 512), (1, 512), 0), out=buf15)
        del arg75_1
        buf18 = reinterpret_tensor(buf14, (1024, 512), (512, 1), 0); del buf14  # reuse
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (1024, 512), (512, 1), 0), reinterpret_tensor(arg33_1, (512, 512), (1, 512), 0), out=buf18)
        del arg33_1
        buf19 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (1024, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 512), (1, 512), 0), out=buf19)
        del arg34_1
        buf20 = reinterpret_tensor(buf9, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf9  # reuse
        # Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf19, (8, 64, 1024), (64, 1, 512), 0), out=buf20)
        buf24 = buf6; del buf6  # reuse
        # Source Nodes: [softmax], Original ATen: [aten._softmax]
        triton_red_fused__softmax_4.run(buf20, arg36_1, buf24, 8192, 1024, grid=grid(8192), stream=stream0)
        buf23 = buf19; del buf19  # reuse
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (1024, 512), (512, 1), 0), reinterpret_tensor(arg35_1, (512, 512), (1, 512), 0), out=buf23)
        del arg35_1
        buf25 = reinterpret_tensor(buf17, (8, 1024, 64), (65536, 64, 1), 0); del buf17  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf23, (8, 1024, 64), (64, 512, 1), 0), out=buf25)
        buf26 = reinterpret_tensor(buf23, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf23  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf25, buf26, 524288, grid=grid(524288), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (1024, 512), (512, 1), 0); del buf25  # reuse
        # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1024, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 512), (1, 512), 0), out=buf27)
        del arg37_1
        buf29 = reinterpret_tensor(buf26, (1, 1024, 512), (524288, 512, 1), 0); del buf26  # reuse
        # Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, inputs_embeds, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5.run(arg132_1, arg32_1, buf27, arg1_1, buf29, 1024, 512, grid=grid(1024), stream=stream0)
        del arg1_1
        buf30 = empty((1024, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1024, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 2048), (1, 512), 0), out=buf30)
        del arg38_1
        buf31 = reinterpret_tensor(buf30, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf30  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf31, 2097152, grid=grid(2097152), stream=stream0)
        buf32 = reinterpret_tensor(buf29, (1024, 512), (512, 1), 0); del buf29  # reuse
        # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg39_1, (2048, 512), (1, 2048), 0), out=buf32)
        del arg39_1
        buf34 = reinterpret_tensor(buf18, (1, 1024, 512), (524288, 512, 1), 0); del buf18  # reuse
        # Source Nodes: [add_7, hidden_states_13, hidden_states_14, hidden_states_5, inputs_embeds, normed_hidden_states_1, pow_3, rsqrt_2, variance_2], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7.run(arg132_1, arg32_1, buf27, buf32, arg2_1, buf34, 1024, 512, grid=grid(1024), stream=stream0)
        del arg2_1
        buf35 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (1024, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 512), (1, 512), 0), out=buf35)
        del arg40_1
        buf36 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (1024, 512), (512, 1), 0), reinterpret_tensor(arg41_1, (512, 512), (1, 512), 0), out=buf36)
        del arg41_1
        buf37 = reinterpret_tensor(buf24, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf24  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf35, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf36, (8, 64, 1024), (64, 1, 512), 0), out=buf37)
        buf41 = reinterpret_tensor(buf20, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf20  # reuse
        # Source Nodes: [softmax_1], Original ATen: [aten._softmax]
        triton_red_fused__softmax_4.run(buf37, arg36_1, buf41, 8192, 1024, grid=grid(8192), stream=stream0)
        buf40 = buf36; del buf36  # reuse
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (1024, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 512), (1, 512), 0), out=buf40)
        del arg42_1
        buf42 = reinterpret_tensor(buf34, (8, 1024, 64), (65536, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf40, (8, 1024, 64), (64, 512, 1), 0), out=buf42)
        buf43 = reinterpret_tensor(buf40, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf40  # reuse
        # Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf42, buf43, 524288, grid=grid(524288), stream=stream0)
        buf44 = reinterpret_tensor(buf42, (1024, 512), (512, 1), 0); del buf42  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (1024, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 512), (1, 512), 0), out=buf44)
        del arg43_1
        buf45 = reinterpret_tensor(buf27, (1, 1024, 512), (524288, 512, 1), 0); del buf27  # reuse
        buf47 = reinterpret_tensor(buf43, (1, 1024, 512), (524288, 512, 1), 0); del buf43  # reuse
        # Source Nodes: [add_9, forwarded_states_2, hidden_states_13, hidden_states_18, hidden_states_19, hidden_states_5, inputs_embeds, pow_4, rsqrt_3, variance_3], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8.run(buf45, arg132_1, arg32_1, buf32, buf44, arg3_1, buf47, 1024, 512, grid=grid(1024), stream=stream0)
        del arg132_1
        del arg3_1
        buf48 = reinterpret_tensor(buf31, (1024, 2048), (2048, 1), 0); del buf31  # reuse
        # Source Nodes: [hidden_states_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1024, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 2048), (1, 512), 0), out=buf48)
        del arg44_1
        buf49 = reinterpret_tensor(buf48, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf48  # reuse
        # Source Nodes: [hidden_states_21], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf49, 2097152, grid=grid(2097152), stream=stream0)
        buf50 = reinterpret_tensor(buf47, (1024, 512), (512, 1), 0); del buf47  # reuse
        # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg45_1, (2048, 512), (1, 2048), 0), out=buf50)
        del arg45_1
        buf52 = reinterpret_tensor(buf44, (1, 1024, 512), (524288, 512, 1), 0); del buf44  # reuse
        # Source Nodes: [add_11, hidden_states_26, hidden_states_27, normed_hidden_states_2, pow_5, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf45, buf50, arg4_1, buf52, 1024, 512, grid=grid(1024), stream=stream0)
        del arg4_1
        buf53 = buf32; del buf32  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (1024, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 512), (1, 512), 0), out=buf53)
        del arg46_1
        buf54 = buf35; del buf35  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (1024, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 512), (1, 512), 0), out=buf54)
        del arg47_1
        buf55 = reinterpret_tensor(buf41, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf41  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf54, (8, 64, 1024), (64, 1, 512), 0), out=buf55)
        buf59 = reinterpret_tensor(buf37, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf37  # reuse
        # Source Nodes: [softmax_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_4.run(buf55, arg36_1, buf59, 8192, 1024, grid=grid(8192), stream=stream0)
        buf58 = buf54; del buf54  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (1024, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 512), (1, 512), 0), out=buf58)
        del arg48_1
        buf60 = reinterpret_tensor(buf52, (8, 1024, 64), (65536, 64, 1), 0); del buf52  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf58, (8, 1024, 64), (64, 512, 1), 0), out=buf60)
        buf61 = reinterpret_tensor(buf58, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf58  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf60, buf61, 524288, grid=grid(524288), stream=stream0)
        buf62 = reinterpret_tensor(buf60, (1024, 512), (512, 1), 0); del buf60  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1024, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 512), (1, 512), 0), out=buf62)
        del arg49_1
        buf64 = reinterpret_tensor(buf61, (1, 1024, 512), (524288, 512, 1), 0); del buf61  # reuse
        # Source Nodes: [add_13, forwarded_states_4, hidden_states_26, hidden_states_31, hidden_states_32, pow_6, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf45, buf50, buf62, arg5_1, buf64, 1024, 512, grid=grid(1024), stream=stream0)
        del arg5_1
        buf65 = reinterpret_tensor(buf49, (1024, 2048), (2048, 1), 0); del buf49  # reuse
        # Source Nodes: [hidden_states_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (1024, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 2048), (1, 512), 0), out=buf65)
        del arg50_1
        buf66 = reinterpret_tensor(buf65, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf65  # reuse
        # Source Nodes: [hidden_states_34], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf66, 2097152, grid=grid(2097152), stream=stream0)
        buf67 = reinterpret_tensor(buf64, (1024, 512), (512, 1), 0); del buf64  # reuse
        # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg51_1, (2048, 512), (1, 2048), 0), out=buf67)
        del arg51_1
        buf69 = reinterpret_tensor(buf53, (1, 1024, 512), (524288, 512, 1), 0); del buf53  # reuse
        # Source Nodes: [add_15, hidden_states_26, hidden_states_31, hidden_states_39, hidden_states_40, normed_hidden_states_3, pow_7, rsqrt_6, variance_6], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf45, buf50, buf62, buf67, arg6_1, buf69, 1024, 512, grid=grid(1024), stream=stream0)
        del arg6_1
        buf70 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 512), (1, 512), 0), out=buf70)
        del arg52_1
        buf71 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 512), (1, 512), 0), out=buf71)
        del arg53_1
        buf72 = reinterpret_tensor(buf59, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf59  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf71, (8, 64, 1024), (64, 1, 512), 0), out=buf72)
        buf76 = reinterpret_tensor(buf55, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf55  # reuse
        # Source Nodes: [softmax_3], Original ATen: [aten._softmax]
        triton_red_fused__softmax_4.run(buf72, arg36_1, buf76, 8192, 1024, grid=grid(8192), stream=stream0)
        buf75 = buf71; del buf71  # reuse
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 512), (1, 512), 0), out=buf75)
        del arg54_1
        buf77 = reinterpret_tensor(buf69, (8, 1024, 64), (65536, 64, 1), 0); del buf69  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf75, (8, 1024, 64), (64, 512, 1), 0), out=buf77)
        buf78 = reinterpret_tensor(buf75, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf75  # reuse
        # Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf77, buf78, 524288, grid=grid(524288), stream=stream0)
        buf79 = reinterpret_tensor(buf77, (1024, 512), (512, 1), 0); del buf77  # reuse
        # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (1024, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 512), (1, 512), 0), out=buf79)
        del arg55_1
        buf80 = buf45; del buf45  # reuse
        buf82 = reinterpret_tensor(buf78, (1, 1024, 512), (524288, 512, 1), 0); del buf78  # reuse
        # Source Nodes: [add_17, forwarded_states_6, hidden_states_26, hidden_states_31, hidden_states_39, hidden_states_44, hidden_states_45, pow_8, rsqrt_7, variance_7], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf80, buf50, buf62, buf67, buf79, arg7_1, buf82, 1024, 512, grid=grid(1024), stream=stream0)
        del arg7_1
        buf83 = reinterpret_tensor(buf66, (1024, 2048), (2048, 1), 0); del buf66  # reuse
        # Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 2048), (1, 512), 0), out=buf83)
        del arg56_1
        buf84 = reinterpret_tensor(buf83, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf83  # reuse
        # Source Nodes: [hidden_states_47], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf84, 2097152, grid=grid(2097152), stream=stream0)
        buf85 = reinterpret_tensor(buf82, (1024, 512), (512, 1), 0); del buf82  # reuse
        # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg57_1, (2048, 512), (1, 2048), 0), out=buf85)
        del arg57_1
        buf87 = reinterpret_tensor(buf79, (1, 1024, 512), (524288, 512, 1), 0); del buf79  # reuse
        # Source Nodes: [add_19, hidden_states_52, hidden_states_53, normed_hidden_states_4, pow_9, rsqrt_8, variance_8], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf80, buf85, arg8_1, buf87, 1024, 512, grid=grid(1024), stream=stream0)
        del arg8_1
        buf88 = buf67; del buf67  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (1024, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 512), (1, 512), 0), out=buf88)
        del arg58_1
        buf89 = buf62; del buf62  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (1024, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 512), (1, 512), 0), out=buf89)
        del arg59_1
        buf90 = reinterpret_tensor(buf76, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf76  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf89, (8, 64, 1024), (64, 1, 512), 0), out=buf90)
        buf94 = reinterpret_tensor(buf72, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf72  # reuse
        # Source Nodes: [softmax_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_4.run(buf90, arg36_1, buf94, 8192, 1024, grid=grid(8192), stream=stream0)
        buf93 = buf89; del buf89  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (1024, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 512), (1, 512), 0), out=buf93)
        del arg60_1
        buf95 = reinterpret_tensor(buf87, (8, 1024, 64), (65536, 64, 1), 0); del buf87  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf93, (8, 1024, 64), (64, 512, 1), 0), out=buf95)
        buf96 = reinterpret_tensor(buf93, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf93  # reuse
        # Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf95, buf96, 524288, grid=grid(524288), stream=stream0)
        buf97 = reinterpret_tensor(buf95, (1024, 512), (512, 1), 0); del buf95  # reuse
        # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (1024, 512), (512, 1), 0), reinterpret_tensor(arg61_1, (512, 512), (1, 512), 0), out=buf97)
        del arg61_1
        buf99 = reinterpret_tensor(buf96, (1, 1024, 512), (524288, 512, 1), 0); del buf96  # reuse
        # Source Nodes: [add_21, forwarded_states_8, hidden_states_52, hidden_states_57, hidden_states_58, pow_10, rsqrt_9, variance_9], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf80, buf85, buf97, arg9_1, buf99, 1024, 512, grid=grid(1024), stream=stream0)
        del arg9_1
        buf100 = reinterpret_tensor(buf84, (1024, 2048), (2048, 1), 0); del buf84  # reuse
        # Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (1024, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 2048), (1, 512), 0), out=buf100)
        del arg62_1
        buf101 = reinterpret_tensor(buf100, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf100  # reuse
        # Source Nodes: [hidden_states_60], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf101, 2097152, grid=grid(2097152), stream=stream0)
        buf102 = reinterpret_tensor(buf99, (1024, 512), (512, 1), 0); del buf99  # reuse
        # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 512), (1, 2048), 0), out=buf102)
        del arg63_1
        buf104 = reinterpret_tensor(buf88, (1, 1024, 512), (524288, 512, 1), 0); del buf88  # reuse
        # Source Nodes: [add_23, hidden_states_52, hidden_states_57, hidden_states_65, hidden_states_66, normed_hidden_states_5, pow_11, rsqrt_10, variance_10], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf80, buf85, buf97, buf102, arg10_1, buf104, 1024, 512, grid=grid(1024), stream=stream0)
        del arg10_1
        buf105 = buf50; del buf50  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (1024, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 512), (1, 512), 0), out=buf105)
        del arg64_1
        buf106 = buf70; del buf70  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (1024, 512), (512, 1), 0), reinterpret_tensor(arg65_1, (512, 512), (1, 512), 0), out=buf106)
        del arg65_1
        buf107 = reinterpret_tensor(buf94, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf94  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf106, (8, 64, 1024), (64, 1, 512), 0), out=buf107)
        buf111 = reinterpret_tensor(buf90, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf90  # reuse
        # Source Nodes: [softmax_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_4.run(buf107, arg36_1, buf111, 8192, 1024, grid=grid(8192), stream=stream0)
        del arg36_1
        buf110 = buf106; del buf106  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (1024, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 512), (1, 512), 0), out=buf110)
        del arg66_1
        buf112 = reinterpret_tensor(buf104, (8, 1024, 64), (65536, 64, 1), 0); del buf104  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf110, (8, 1024, 64), (64, 512, 1), 0), out=buf112)
        buf113 = reinterpret_tensor(buf110, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf110  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf112, buf113, 524288, grid=grid(524288), stream=stream0)
        buf114 = reinterpret_tensor(buf112, (1024, 512), (512, 1), 0); del buf112  # reuse
        # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (1024, 512), (512, 1), 0), reinterpret_tensor(arg67_1, (512, 512), (1, 512), 0), out=buf114)
        del arg67_1
        buf115 = reinterpret_tensor(buf102, (1, 1024, 512), (524288, 512, 1), 0); del buf102  # reuse
        buf117 = reinterpret_tensor(buf113, (1, 1024, 512), (524288, 512, 1), 0); del buf113  # reuse
        # Source Nodes: [add_25, forwarded_states_10, hidden_states_52, hidden_states_57, hidden_states_65, hidden_states_70, hidden_states_71, pow_12, rsqrt_11, variance_11], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf115, buf80, buf85, buf97, buf114, arg11_1, buf117, 1024, 512, grid=grid(1024), stream=stream0)
        del arg11_1
        buf118 = reinterpret_tensor(buf101, (1024, 2048), (2048, 1), 0); del buf101  # reuse
        # Source Nodes: [hidden_states_72], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (1024, 512), (512, 1), 0), reinterpret_tensor(arg68_1, (512, 2048), (1, 512), 0), out=buf118)
        del arg68_1
        buf119 = reinterpret_tensor(buf118, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf118  # reuse
        # Source Nodes: [hidden_states_73], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf119, 2097152, grid=grid(2097152), stream=stream0)
        buf120 = reinterpret_tensor(buf117, (1024, 512), (512, 1), 0); del buf117  # reuse
        # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg69_1, (2048, 512), (1, 2048), 0), out=buf120)
        del arg69_1
        buf122 = reinterpret_tensor(buf97, (1, 1024, 512), (524288, 512, 1), 0); del buf97  # reuse
        # Source Nodes: [add_27, hidden_states_78, hidden_states_79, hidden_states_80, pow_13, rsqrt_12, variance_12], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf115, buf120, arg12_1, buf122, 1024, 512, grid=grid(1024), stream=stream0)
        del arg12_1
        buf123 = buf120; del buf120  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 512), (1, 512), 0), out=buf123)
        del arg76_1
        buf124 = reinterpret_tensor(buf111, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf111  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf123, (8, 64, 1024), (64, 1, 512), 0), out=buf124)
        buf128 = reinterpret_tensor(buf107, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf107  # reuse
        # Source Nodes: [softmax_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf124, buf128, 8192, 1024, grid=grid(8192), stream=stream0)
        buf127 = buf15; del buf15  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 512), (1, 512), 0), out=buf127)
        del arg77_1
        buf129 = reinterpret_tensor(buf115, (8, 1024, 64), (65536, 64, 1), 0); del buf115  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf128, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf127, (8, 1024, 64), (64, 512, 1), 0), out=buf129)
        buf130 = reinterpret_tensor(buf85, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf85  # reuse
        # Source Nodes: [contiguous_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf129, buf130, 524288, grid=grid(524288), stream=stream0)
        buf131 = reinterpret_tensor(buf129, (1024, 512), (512, 1), 0); del buf129  # reuse
        # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (1024, 512), (512, 1), 0), reinterpret_tensor(arg78_1, (512, 512), (1, 512), 0), out=buf131)
        del arg78_1
        buf133 = reinterpret_tensor(buf130, (1, 1024, 512), (524288, 512, 1), 0); del buf130  # reuse
        # Source Nodes: [add_36, forwarded_states_12, hidden_states_88, hidden_states_92, hidden_states_93, inputs_embeds_1, pow_16, rsqrt_15, variance_15], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7.run(arg134_1, arg32_1, buf12, buf131, arg15_1, buf133, 1024, 512, grid=grid(1024), stream=stream0)
        del arg15_1
        buf134 = reinterpret_tensor(buf119, (1024, 2048), (2048, 1), 0); del buf119  # reuse
        # Source Nodes: [hidden_states_94], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (1024, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 2048), (1, 512), 0), out=buf134)
        del arg79_1
        buf135 = reinterpret_tensor(buf134, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf134  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf135, 2097152, grid=grid(2097152), stream=stream0)
        buf136 = reinterpret_tensor(buf133, (1024, 512), (512, 1), 0); del buf133  # reuse
        # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg80_1, (2048, 512), (1, 2048), 0), out=buf136)
        del arg80_1
        buf137 = reinterpret_tensor(buf12, (1, 1024, 512), (524288, 512, 1), 0); del buf12  # reuse
        buf139 = buf80; del buf80  # reuse
        # Source Nodes: [add_38, hidden_states_100, hidden_states_101, hidden_states_88, hidden_states_92, inputs_embeds_1, normed_hidden_states_8, pow_17, rsqrt_16, variance_16], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8.run(buf137, arg134_1, arg32_1, buf131, buf136, arg16_1, buf139, 1024, 512, grid=grid(1024), stream=stream0)
        del arg134_1
        del arg16_1
        del arg32_1
        buf140 = buf136; del buf136  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (1024, 512), (512, 1), 0), reinterpret_tensor(arg81_1, (512, 512), (1, 512), 0), out=buf140)
        del arg81_1
        buf141 = buf131; del buf131  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (1024, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 512), (1, 512), 0), out=buf141)
        del arg82_1
        buf142 = reinterpret_tensor(buf128, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf128  # reuse
        # Source Nodes: [scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf140, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf141, (8, 64, 1024), (64, 1, 512), 0), out=buf142)
        buf144 = reinterpret_tensor(buf124, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf124  # reuse
        buf147 = reinterpret_tensor(buf4, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf4  # reuse
        # Source Nodes: [softmax_8], Original ATen: [aten._softmax]
        triton_red_fused__softmax_1.run(buf142, arg73_1, buf144, buf147, 8192, 1024, grid=grid(8192), stream=stream0)
        buf146 = buf140; del buf140  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (1024, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 512), (1, 512), 0), out=buf146)
        del arg83_1
        buf148 = reinterpret_tensor(buf139, (8, 1024, 64), (65536, 64, 1), 0); del buf139  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf147, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf146, (8, 1024, 64), (64, 512, 1), 0), out=buf148)
        buf149 = reinterpret_tensor(buf114, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf114  # reuse
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf148, buf149, 524288, grid=grid(524288), stream=stream0)
        buf150 = reinterpret_tensor(buf148, (1024, 512), (512, 1), 0); del buf148  # reuse
        # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (1024, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 512), (1, 512), 0), out=buf150)
        del arg84_1
        buf152 = reinterpret_tensor(buf149, (1, 1024, 512), (524288, 512, 1), 0); del buf149  # reuse
        # Source Nodes: [add_40, hidden_states_105, hidden_states_106, normed_hidden_states_9, pow_18, rsqrt_17, variance_17], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf137, buf150, arg17_1, buf152, 1024, 512, grid=grid(1024), stream=stream0)
        del arg17_1
        buf153 = buf105; del buf105  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (1024, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 512), (1, 512), 0), out=buf153)
        del arg85_1
        buf154 = reinterpret_tensor(buf152, (1024, 512), (512, 1), 0); del buf152  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), out=buf154)
        del arg86_1
        buf155 = reinterpret_tensor(buf147, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf147  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf154, (8, 64, 1024), (64, 1, 512), 0), out=buf155)
        buf159 = buf144; del buf144  # reuse
        # Source Nodes: [softmax_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf155, buf159, 8192, 1024, grid=grid(8192), stream=stream0)
        buf158 = buf153; del buf153  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 512), (1, 512), 0), out=buf158)
        del arg87_1
        buf160 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf159, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf158, (8, 1024, 64), (64, 512, 1), 0), out=buf160)
        buf161 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf160, buf161, 524288, grid=grid(524288), stream=stream0)
        buf162 = reinterpret_tensor(buf160, (1024, 512), (512, 1), 0); del buf160  # reuse
        # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (1024, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 512), (1, 512), 0), out=buf162)
        del arg88_1
        buf164 = reinterpret_tensor(buf161, (1, 1024, 512), (524288, 512, 1), 0); del buf161  # reuse
        # Source Nodes: [add_42, forwarded_states_14, hidden_states_105, hidden_states_109, hidden_states_110, pow_19, rsqrt_18, variance_18], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf137, buf150, buf162, arg18_1, buf164, 1024, 512, grid=grid(1024), stream=stream0)
        del arg18_1
        buf165 = reinterpret_tensor(buf135, (1024, 2048), (2048, 1), 0); del buf135  # reuse
        # Source Nodes: [hidden_states_111], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (1024, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 2048), (1, 512), 0), out=buf165)
        del arg89_1
        buf166 = reinterpret_tensor(buf165, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf165  # reuse
        # Source Nodes: [hidden_states_112], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf166, 2097152, grid=grid(2097152), stream=stream0)
        buf167 = reinterpret_tensor(buf164, (1024, 512), (512, 1), 0); del buf164  # reuse
        # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg90_1, (2048, 512), (1, 2048), 0), out=buf167)
        del arg90_1
        buf169 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, hidden_states_105, hidden_states_109, hidden_states_117, hidden_states_118, normed_hidden_states_10, pow_20, rsqrt_19, variance_19], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf137, buf150, buf162, buf167, arg19_1, buf169, 1024, 512, grid=grid(1024), stream=stream0)
        del arg19_1
        buf170 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 512), (1, 512), 0), out=buf170)
        del arg91_1
        buf171 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 512), (512, 1), 0), reinterpret_tensor(arg92_1, (512, 512), (1, 512), 0), out=buf171)
        del arg92_1
        buf172 = reinterpret_tensor(buf159, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf159  # reuse
        # Source Nodes: [scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf170, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf171, (8, 64, 1024), (64, 1, 512), 0), out=buf172)
        buf174 = reinterpret_tensor(buf155, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf155  # reuse
        buf177 = reinterpret_tensor(buf142, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf142  # reuse
        # Source Nodes: [softmax_10], Original ATen: [aten._softmax]
        triton_red_fused__softmax_1.run(buf172, arg73_1, buf174, buf177, 8192, 1024, grid=grid(8192), stream=stream0)
        buf176 = buf170; del buf170  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 512), (1, 512), 0), out=buf176)
        del arg93_1
        buf178 = reinterpret_tensor(buf169, (8, 1024, 64), (65536, 64, 1), 0); del buf169  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf177, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf176, (8, 1024, 64), (64, 512, 1), 0), out=buf178)
        buf179 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf178, buf179, 524288, grid=grid(524288), stream=stream0)
        buf180 = reinterpret_tensor(buf178, (1024, 512), (512, 1), 0); del buf178  # reuse
        # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (1024, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 512), (1, 512), 0), out=buf180)
        del arg94_1
        buf181 = buf137; del buf137  # reuse
        buf183 = reinterpret_tensor(buf179, (1, 1024, 512), (524288, 512, 1), 0); del buf179  # reuse
        # Source Nodes: [add_46, hidden_states_105, hidden_states_109, hidden_states_117, hidden_states_122, hidden_states_123, normed_hidden_states_11, pow_21, rsqrt_20, variance_20], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf181, buf150, buf162, buf167, buf180, arg20_1, buf183, 1024, 512, grid=grid(1024), stream=stream0)
        del arg20_1
        buf184 = buf180; del buf180  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (1024, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 512), (1, 512), 0), out=buf184)
        del arg95_1
        buf185 = reinterpret_tensor(buf183, (1024, 512), (512, 1), 0); del buf183  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 512), (1, 512), 0), out=buf185)
        del arg96_1
        buf186 = reinterpret_tensor(buf177, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf177  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf185, (8, 64, 1024), (64, 1, 512), 0), out=buf186)
        buf190 = buf174; del buf174  # reuse
        # Source Nodes: [softmax_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf186, buf190, 8192, 1024, grid=grid(8192), stream=stream0)
        buf189 = buf184; del buf184  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg97_1, (512, 512), (1, 512), 0), out=buf189)
        del arg97_1
        buf191 = reinterpret_tensor(buf167, (8, 1024, 64), (65536, 64, 1), 0); del buf167  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf190, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf189, (8, 1024, 64), (64, 512, 1), 0), out=buf191)
        buf192 = reinterpret_tensor(buf162, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf162  # reuse
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf191, buf192, 524288, grid=grid(524288), stream=stream0)
        buf193 = reinterpret_tensor(buf191, (1024, 512), (512, 1), 0); del buf191  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (1024, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 512), (1, 512), 0), out=buf193)
        del arg98_1
        buf195 = reinterpret_tensor(buf192, (1, 1024, 512), (524288, 512, 1), 0); del buf192  # reuse
        # Source Nodes: [add_48, forwarded_states_16, hidden_states_126, hidden_states_127, pow_22, rsqrt_21, variance_21], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf181, buf193, arg21_1, buf195, 1024, 512, grid=grid(1024), stream=stream0)
        del arg21_1
        buf196 = reinterpret_tensor(buf166, (1024, 2048), (2048, 1), 0); del buf166  # reuse
        # Source Nodes: [hidden_states_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 512), (512, 1), 0), reinterpret_tensor(arg99_1, (512, 2048), (1, 512), 0), out=buf196)
        del arg99_1
        buf197 = reinterpret_tensor(buf196, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf196  # reuse
        # Source Nodes: [hidden_states_129], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf197, 2097152, grid=grid(2097152), stream=stream0)
        buf198 = reinterpret_tensor(buf195, (1024, 512), (512, 1), 0); del buf195  # reuse
        # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg100_1, (2048, 512), (1, 2048), 0), out=buf198)
        del arg100_1
        buf200 = reinterpret_tensor(buf150, (1, 1024, 512), (524288, 512, 1), 0); del buf150  # reuse
        # Source Nodes: [add_50, hidden_states_126, hidden_states_134, hidden_states_135, normed_hidden_states_12, pow_23, rsqrt_22, variance_22], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf181, buf193, buf198, arg22_1, buf200, 1024, 512, grid=grid(1024), stream=stream0)
        del arg22_1
        buf201 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (1024, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 512), (1, 512), 0), out=buf201)
        del arg101_1
        buf202 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (1024, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 512), (1, 512), 0), out=buf202)
        del arg102_1
        buf203 = reinterpret_tensor(buf190, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf190  # reuse
        # Source Nodes: [scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf202, (8, 64, 1024), (64, 1, 512), 0), out=buf203)
        buf205 = reinterpret_tensor(buf186, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf186  # reuse
        buf208 = reinterpret_tensor(buf172, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf172  # reuse
        # Source Nodes: [softmax_12], Original ATen: [aten._softmax]
        triton_red_fused__softmax_1.run(buf203, arg73_1, buf205, buf208, 8192, 1024, grid=grid(8192), stream=stream0)
        buf207 = buf201; del buf201  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (1024, 512), (512, 1), 0), reinterpret_tensor(arg103_1, (512, 512), (1, 512), 0), out=buf207)
        del arg103_1
        buf209 = reinterpret_tensor(buf200, (8, 1024, 64), (65536, 64, 1), 0); del buf200  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf207, (8, 1024, 64), (64, 512, 1), 0), out=buf209)
        buf210 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf209, buf210, 524288, grid=grid(524288), stream=stream0)
        buf211 = reinterpret_tensor(buf209, (1024, 512), (512, 1), 0); del buf209  # reuse
        # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (1024, 512), (512, 1), 0), reinterpret_tensor(arg104_1, (512, 512), (1, 512), 0), out=buf211)
        del arg104_1
        buf213 = reinterpret_tensor(buf210, (1, 1024, 512), (524288, 512, 1), 0); del buf210  # reuse
        # Source Nodes: [add_52, hidden_states_126, hidden_states_134, hidden_states_139, hidden_states_140, normed_hidden_states_13, pow_24, rsqrt_23, variance_23], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf181, buf193, buf198, buf211, arg23_1, buf213, 1024, 512, grid=grid(1024), stream=stream0)
        del arg23_1
        buf214 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (1024, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 512), (1, 512), 0), out=buf214)
        del arg105_1
        buf215 = reinterpret_tensor(buf213, (1024, 512), (512, 1), 0); del buf213  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 512), (1, 512), 0), out=buf215)
        del arg106_1
        buf216 = reinterpret_tensor(buf208, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf208  # reuse
        # Source Nodes: [scores_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf214, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf215, (8, 64, 1024), (64, 1, 512), 0), out=buf216)
        buf220 = buf205; del buf205  # reuse
        # Source Nodes: [softmax_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf216, buf220, 8192, 1024, grid=grid(8192), stream=stream0)
        buf219 = buf214; del buf214  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 512), (1, 512), 0), out=buf219)
        del arg107_1
        buf221 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf219, (8, 1024, 64), (64, 512, 1), 0), out=buf221)
        buf222 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf221, buf222, 524288, grid=grid(524288), stream=stream0)
        buf223 = reinterpret_tensor(buf221, (1024, 512), (512, 1), 0); del buf221  # reuse
        # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (1024, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 512), (1, 512), 0), out=buf223)
        del arg108_1
        buf224 = buf181; del buf181  # reuse
        buf226 = reinterpret_tensor(buf222, (1, 1024, 512), (524288, 512, 1), 0); del buf222  # reuse
        # Source Nodes: [add_54, forwarded_states_18, hidden_states_126, hidden_states_134, hidden_states_139, hidden_states_143, hidden_states_144, pow_25, rsqrt_24, variance_24], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf224, buf193, buf198, buf211, buf223, arg24_1, buf226, 1024, 512, grid=grid(1024), stream=stream0)
        del arg24_1
        buf227 = reinterpret_tensor(buf197, (1024, 2048), (2048, 1), 0); del buf197  # reuse
        # Source Nodes: [hidden_states_145], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (1024, 512), (512, 1), 0), reinterpret_tensor(arg109_1, (512, 2048), (1, 512), 0), out=buf227)
        del arg109_1
        buf228 = reinterpret_tensor(buf227, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf227  # reuse
        # Source Nodes: [hidden_states_146], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf228, 2097152, grid=grid(2097152), stream=stream0)
        buf229 = reinterpret_tensor(buf226, (1024, 512), (512, 1), 0); del buf226  # reuse
        # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 512), (1, 2048), 0), out=buf229)
        del arg110_1
        buf231 = reinterpret_tensor(buf223, (1, 1024, 512), (524288, 512, 1), 0); del buf223  # reuse
        # Source Nodes: [add_56, hidden_states_151, hidden_states_152, normed_hidden_states_14, pow_26, rsqrt_25, variance_25], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf224, buf229, arg25_1, buf231, 1024, 512, grid=grid(1024), stream=stream0)
        del arg25_1
        buf232 = buf211; del buf211  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 512), (1, 512), 0), out=buf232)
        del arg111_1
        buf233 = buf198; del buf198  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 512), (1, 512), 0), out=buf233)
        del arg112_1
        buf234 = reinterpret_tensor(buf220, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf220  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf233, (8, 64, 1024), (64, 1, 512), 0), out=buf234)
        buf236 = reinterpret_tensor(buf216, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf216  # reuse
        buf239 = reinterpret_tensor(buf203, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf203  # reuse
        # Source Nodes: [softmax_14], Original ATen: [aten._softmax]
        triton_red_fused__softmax_1.run(buf234, arg73_1, buf236, buf239, 8192, 1024, grid=grid(8192), stream=stream0)
        buf238 = buf232; del buf232  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 512), (1, 512), 0), out=buf238)
        del arg113_1
        buf240 = reinterpret_tensor(buf231, (8, 1024, 64), (65536, 64, 1), 0); del buf231  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf238, (8, 1024, 64), (64, 512, 1), 0), out=buf240)
        buf241 = reinterpret_tensor(buf193, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf193  # reuse
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf240, buf241, 524288, grid=grid(524288), stream=stream0)
        buf242 = reinterpret_tensor(buf240, (1024, 512), (512, 1), 0); del buf240  # reuse
        # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (1024, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 512), (1, 512), 0), out=buf242)
        del arg114_1
        buf244 = reinterpret_tensor(buf241, (1, 1024, 512), (524288, 512, 1), 0); del buf241  # reuse
        # Source Nodes: [add_58, hidden_states_151, hidden_states_156, hidden_states_157, normed_hidden_states_15, pow_27, rsqrt_26, variance_26], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf224, buf229, buf242, arg26_1, buf244, 1024, 512, grid=grid(1024), stream=stream0)
        del arg26_1
        buf245 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (1024, 512), (512, 1), 0), reinterpret_tensor(arg115_1, (512, 512), (1, 512), 0), out=buf245)
        del arg115_1
        buf246 = reinterpret_tensor(buf244, (1024, 512), (512, 1), 0); del buf244  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 512), (1, 512), 0), out=buf246)
        del arg116_1
        buf247 = reinterpret_tensor(buf239, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf239  # reuse
        # Source Nodes: [scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf245, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf246, (8, 64, 1024), (64, 1, 512), 0), out=buf247)
        buf251 = buf236; del buf236  # reuse
        # Source Nodes: [softmax_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf247, buf251, 8192, 1024, grid=grid(8192), stream=stream0)
        buf250 = buf245; del buf245  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 512), (1, 512), 0), out=buf250)
        del arg117_1
        buf252 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf251, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf250, (8, 1024, 64), (64, 512, 1), 0), out=buf252)
        buf253 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf252, buf253, 524288, grid=grid(524288), stream=stream0)
        buf254 = reinterpret_tensor(buf252, (1024, 512), (512, 1), 0); del buf252  # reuse
        # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (1024, 512), (512, 1), 0), reinterpret_tensor(arg118_1, (512, 512), (1, 512), 0), out=buf254)
        del arg118_1
        buf256 = reinterpret_tensor(buf253, (1, 1024, 512), (524288, 512, 1), 0); del buf253  # reuse
        # Source Nodes: [add_60, forwarded_states_20, hidden_states_151, hidden_states_156, hidden_states_160, hidden_states_161, pow_28, rsqrt_27, variance_27], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf224, buf229, buf242, buf254, arg27_1, buf256, 1024, 512, grid=grid(1024), stream=stream0)
        del arg27_1
        buf257 = reinterpret_tensor(buf228, (1024, 2048), (2048, 1), 0); del buf228  # reuse
        # Source Nodes: [hidden_states_162], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (1024, 512), (512, 1), 0), reinterpret_tensor(arg119_1, (512, 2048), (1, 512), 0), out=buf257)
        del arg119_1
        buf258 = reinterpret_tensor(buf257, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf257  # reuse
        # Source Nodes: [hidden_states_163], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf258, 2097152, grid=grid(2097152), stream=stream0)
        buf259 = reinterpret_tensor(buf256, (1024, 512), (512, 1), 0); del buf256  # reuse
        # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg120_1, (2048, 512), (1, 2048), 0), out=buf259)
        del arg120_1
        buf260 = buf224; del buf224  # reuse
        buf262 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_62, hidden_states_151, hidden_states_156, hidden_states_160, hidden_states_168, hidden_states_169, normed_hidden_states_16, pow_29, rsqrt_28, variance_28], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf260, buf229, buf242, buf254, buf259, arg28_1, buf262, 1024, 512, grid=grid(1024), stream=stream0)
        del arg28_1
        buf263 = buf259; del buf259  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (1024, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 512), (1, 512), 0), out=buf263)
        del arg121_1
        buf264 = buf254; del buf254  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (1024, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 512), (1, 512), 0), out=buf264)
        del arg122_1
        buf265 = reinterpret_tensor(buf251, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf251  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf263, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf264, (8, 64, 1024), (64, 1, 512), 0), out=buf265)
        buf267 = reinterpret_tensor(buf247, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf247  # reuse
        buf270 = reinterpret_tensor(buf234, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf234  # reuse
        # Source Nodes: [softmax_16], Original ATen: [aten._softmax]
        triton_red_fused__softmax_1.run(buf265, arg73_1, buf267, buf270, 8192, 1024, grid=grid(8192), stream=stream0)
        del arg73_1
        del buf265
        buf269 = buf263; del buf263  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (1024, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 512), (1, 512), 0), out=buf269)
        del arg123_1
        buf271 = reinterpret_tensor(buf262, (8, 1024, 64), (65536, 64, 1), 0); del buf262  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf269, (8, 1024, 64), (64, 512, 1), 0), out=buf271)
        buf272 = reinterpret_tensor(buf242, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf242  # reuse
        # Source Nodes: [contiguous_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf271, buf272, 524288, grid=grid(524288), stream=stream0)
        buf273 = reinterpret_tensor(buf271, (1024, 512), (512, 1), 0); del buf271  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf272, (1024, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 512), (1, 512), 0), out=buf273)
        del arg124_1
        buf275 = reinterpret_tensor(buf272, (1, 1024, 512), (524288, 512, 1), 0); del buf272  # reuse
        # Source Nodes: [add_64, hidden_states_173, hidden_states_174, normed_hidden_states_17, pow_30, rsqrt_29, variance_29], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf260, buf273, arg29_1, buf275, 1024, 512, grid=grid(1024), stream=stream0)
        del arg29_1
        buf276 = buf229; del buf229  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (1024, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 512), (1, 512), 0), out=buf276)
        del arg125_1
        buf277 = reinterpret_tensor(buf275, (1024, 512), (512, 1), 0); del buf275  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 512), (1, 512), 0), out=buf277)
        del arg126_1
        buf278 = reinterpret_tensor(buf270, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf270  # reuse
        # Source Nodes: [scores_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf276, (8, 1024, 64), (64, 512, 1), 0), reinterpret_tensor(buf277, (8, 64, 1024), (64, 1, 512), 0), out=buf278)
        buf282 = buf267; del buf267  # reuse
        # Source Nodes: [softmax_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf278, buf282, 8192, 1024, grid=grid(8192), stream=stream0)
        del buf278
        buf281 = buf276; del buf276  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 512), (1, 512), 0), out=buf281)
        del arg127_1
        buf283 = empty((8, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf282, (8, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf281, (8, 1024, 64), (64, 512, 1), 0), out=buf283)
        del buf282
        buf284 = empty((1, 1024, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf283, buf284, 524288, grid=grid(524288), stream=stream0)
        buf285 = reinterpret_tensor(buf283, (1024, 512), (512, 1), 0); del buf283  # reuse
        # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (1024, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 512), (1, 512), 0), out=buf285)
        del arg128_1
        buf287 = reinterpret_tensor(buf284, (1, 1024, 512), (524288, 512, 1), 0); del buf284  # reuse
        # Source Nodes: [add_66, forwarded_states_22, hidden_states_173, hidden_states_177, hidden_states_178, pow_31, rsqrt_30, variance_30], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf260, buf273, buf285, arg30_1, buf287, 1024, 512, grid=grid(1024), stream=stream0)
        del arg30_1
        buf288 = reinterpret_tensor(buf258, (1024, 2048), (2048, 1), 0); del buf258  # reuse
        # Source Nodes: [hidden_states_179], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (1024, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 2048), (1, 512), 0), out=buf288)
        del arg129_1
        buf289 = reinterpret_tensor(buf288, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf288  # reuse
        # Source Nodes: [hidden_states_180], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf289, 2097152, grid=grid(2097152), stream=stream0)
        buf290 = reinterpret_tensor(buf287, (1024, 512), (512, 1), 0); del buf287  # reuse
        # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (1024, 2048), (2048, 1), 0), reinterpret_tensor(arg130_1, (2048, 512), (1, 2048), 0), out=buf290)
        del arg130_1
        del buf289
        buf292 = empty((1, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_68, hidden_states_173, hidden_states_177, hidden_states_185, hidden_states_186, hidden_states_187, pow_32, rsqrt_31, sequence_output_1, variance_31], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_15.run(buf260, buf273, buf285, buf290, arg31_1, buf292, 1024, 512, grid=grid(1024), stream=stream0)
        del arg31_1
        del buf260
        del buf273
        del buf285
        del buf290
        buf293 = empty((1024, 32128), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (1024, 512), (512, 1), 0), reinterpret_tensor(arg131_1, (512, 32128), (1, 512), 0), out=buf293)
        del arg131_1
        del buf292
        buf294 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        buf295 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_16.run(buf293, buf294, buf295, 1024, 32128, grid=grid(1024), stream=stream0)
        buf296 = empty((), device='cuda', dtype=torch.float32)
        buf298 = buf296; del buf296  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_17.run(buf298, arg133_1, buf293, buf294, buf295, 1, 1024, grid=grid(1), stream=stream0)
        del arg133_1
        return (buf298, reinterpret_tensor(buf293, (1, 1024, 32128), (32899072, 32128, 1), 0), reinterpret_tensor(buf3, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf8, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf123, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf127, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf141, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf146, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf154, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf158, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf171, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf176, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf185, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf189, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf202, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf207, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf215, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf219, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf233, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf238, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf246, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf250, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf264, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf269, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf277, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), reinterpret_tensor(buf281, (1, 8, 1024, 64), (524288, 64, 512, 1), 0), buf122, )


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
    arg132_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg133_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg134_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('T5ForConditionalGeneration', benchmark_compiled_module)
