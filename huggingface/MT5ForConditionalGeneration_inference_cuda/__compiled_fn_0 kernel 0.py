
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


# kernel path: /tmp/torchinductor_youkaichao/ry/crypd4psjzbax2kuzxzzm4b5c6wrzdtvyqypo37w2nd2l637bubc.py
# Source Nodes: [add_52, hidden_states_102, inputs_embeds_1, normed_hidden_states_8, pow_26, rsqrt_17, variance_17], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_52 => add_61
# hidden_states_102 => mul_80
# inputs_embeds_1 => embedding_2
# normed_hidden_states_8 => mul_81
# pow_26 => pow_26
# rsqrt_17 => rsqrt_17
# variance_17 => mean_17
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp1 = tmp0 + 250112
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 250112")
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
        tmp10 = tmp0 + 250112
        tmp11 = tmp0 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp0)
        tl.device_assert(((0 <= tmp12) & (tmp12 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp12 < 250112")
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


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vylrdazcjxpayx5ptufcshtci4mjb4ttsinise5ree5zgxjx6j.py
# Source Nodes: [softmax_8], Original ATen: [aten._softmax]
# softmax_8 => amax_8, div_12, exp_8, sub_13, sum_9
triton_per_fused__softmax_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
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
    tmp21 = tl.load(in_ptr1 + (x1 + (6*tmp20)), xmask, eviction_policy='evict_last')
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
    tmp34 = tl.where(rmask & xmask, tmp32, float("-inf"))
    tmp35 = triton_helpers.max2(tmp34, 1)[:, None]
    tmp36 = tmp31 - tmp35
    tmp37 = tl.exp(tmp36)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp37 / tmp41
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp42, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/cczuv7mcrlbavf4j5dcd5xztt6ppmgjvd43umgnp7hzfifndhcra.py
# Source Nodes: [contiguous_8], Original ATen: [aten.clone]
# contiguous_8 => clone_44
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 6
    x2 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cgg6kfamolo5xf2qqvcrveknxuxjamq2hyxh4svpf6ecfiwdddsb.py
# Source Nodes: [add, add_57, hidden_states_1, hidden_states_106, hidden_states_107, inputs_embeds, inputs_embeds_1, normed_hidden_states, normed_hidden_states_9, pow_1, pow_27, rsqrt, rsqrt_18, variance, variance_18], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add => add
# add_57 => add_67
# hidden_states_1 => mul_1
# hidden_states_106 => add_66
# hidden_states_107 => mul_83
# inputs_embeds => embedding
# inputs_embeds_1 => embedding_2
# normed_hidden_states => mul_2
# normed_hidden_states_9 => mul_84
# pow_1 => pow_1
# pow_27 => pow_27
# rsqrt => rsqrt
# rsqrt_18 => rsqrt_18
# variance => mean
# variance_18 => mean_18
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp1 = tmp0 + 250112
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 250112")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp11 + 250112
        tmp13 = tmp11 < 0
        tmp14 = tl.where(tmp13, tmp12, tmp11)
        tl.device_assert(((0 <= tmp14) & (tmp14 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp14 < 250112")
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
        tmp21 = tmp0 + 250112
        tmp22 = tmp0 < 0
        tmp23 = tl.where(tmp22, tmp21, tmp0)
        tl.device_assert(((0 <= tmp23) & (tmp23 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp23 < 250112")
        tmp24 = tl.load(in_ptr1 + (r1 + (512*tmp23)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tmp24 + tmp25
        tmp27 = 512.0
        tmp28 = tmp9 / tmp27
        tmp29 = 1e-06
        tmp30 = tmp28 + tmp29
        tmp31 = tl.math.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp33 = tmp20 * tmp32
        tmp35 = tmp11 + 250112
        tmp36 = tmp11 < 0
        tmp37 = tl.where(tmp36, tmp35, tmp11)
        tl.device_assert(((0 <= tmp37) & (tmp37 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp37 < 250112")
        tmp38 = tl.load(in_ptr1 + (r1 + (512*tmp37)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tmp18 / tmp27
        tmp40 = tmp39 + tmp29
        tmp41 = tl.math.rsqrt(tmp40)
        tmp42 = tmp38 * tmp41
        tmp43 = tmp34 * tmp42
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp33, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp43, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3i/c3imcxd53zlgarbsofa7pi5hdwtxhistupglvpkvq4slpuow37tv.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, div_2, exp, sub_2, sum_1
triton_per_fused__softmax_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
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
    tl.device_assert(((0 <= tmp26) & (tmp26 < 32)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp26 < 32")
    tmp27 = tl.load(in_ptr1 + (x1 + (6*tmp26)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp0 + tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, float("-inf"))
    tmp32 = triton_helpers.max2(tmp31, 1)[:, None]
    tmp33 = tmp28 - tmp32
    tmp34 = tl.exp(tmp33)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tmp34 / tmp38
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ra/crapqjj5gmri3mp54v3yi65422wo4iz7g63li4t33qhxqrrhqsdr.py
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tmp0 + 250112
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 250112")
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


# kernel path: /tmp/torchinductor_youkaichao/ry/cryyvfodhx3dzlel4kggw34kdwbhu66cy3ffw2iwjintl6xy3jqn.py
# Source Nodes: [add_6, add_7, hidden_gelu, hidden_states_7, mul_7, mul_8, mul_9, pow_3, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
# add_6 => add_8
# add_7 => add_9
# hidden_gelu => mul_10
# hidden_states_7 => mul_11
# mul_7 => mul_7
# mul_8 => mul_8
# mul_9 => mul_9
# pow_3 => pow_3
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp14 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = 0.044715
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = 0.7978845608028654
    tmp9 = tmp7 * tmp8
    tmp10 = tl.math.tanh(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp2 * tmp12
    tmp15 = tmp13 * tmp14
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cupg5m3cfgnf6ptmw5wz4xeto27xr4j76h6er4sljhlbzp7rkxyc.py
# Source Nodes: [add_9, hidden_states_12, hidden_states_13, hidden_states_5, inputs_embeds, normed_hidden_states_1, pow_4, rsqrt_2, variance_2], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_9 => add_11
# hidden_states_12 => add_10
# hidden_states_13 => mul_12
# hidden_states_5 => add_6
# inputs_embeds => embedding
# normed_hidden_states_1 => mul_13
# pow_4 => pow_4
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tmp0 + 250112
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 250112")
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


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbsngv35pxs6urzbwsyq4v6efglocm3nepkuub4nyhjhknlkzpw.py
# Source Nodes: [add_11, forwarded_states_2, hidden_states_12, hidden_states_17, hidden_states_18, hidden_states_5, inputs_embeds, pow_5, rsqrt_3, variance_3], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_11 => add_14
# forwarded_states_2 => mul_15
# hidden_states_12 => add_10
# hidden_states_17 => add_13
# hidden_states_18 => mul_14
# hidden_states_5 => add_6
# inputs_embeds => embedding
# pow_5 => pow_5
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tmp0 + 250112
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 250112")
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


# kernel path: /tmp/torchinductor_youkaichao/sy/csy65wwwgvjxapmuizflfjd3e5oywxyjx3xk5sr5jf7fudovrxgs.py
# Source Nodes: [add_15, hidden_states_24, hidden_states_25, normed_hidden_states_2, pow_7, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_15 => add_18
# hidden_states_24 => add_17
# hidden_states_25 => mul_21
# normed_hidden_states_2 => mul_22
# pow_7 => pow_7
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6x5bpdewsy7xddza6tarsit2kwbdhhvlzebicfrcrtpbstvx6o.py
# Source Nodes: [add_17, forwarded_states_4, hidden_states_24, hidden_states_29, hidden_states_30, pow_8, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_17 => add_21
# forwarded_states_4 => mul_24
# hidden_states_24 => add_17
# hidden_states_29 => add_20
# hidden_states_30 => mul_23
# pow_8 => pow_8
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/az/cazufq3fooqkdfrepsfkezepqquf5ve3qfv4an4mtzouwuxxil7s.py
# Source Nodes: [add_21, hidden_states_24, hidden_states_29, hidden_states_36, hidden_states_37, normed_hidden_states_3, pow_10, rsqrt_6, variance_6], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_21 => add_25
# hidden_states_24 => add_17
# hidden_states_29 => add_20
# hidden_states_36 => add_24
# hidden_states_37 => mul_30
# normed_hidden_states_3 => mul_31
# pow_10 => pow_10
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4jruch24o4m56rjxk7zjwwgzespook53fmnb57ox27h6a2i4zu.py
# Source Nodes: [add_23, forwarded_states_6, hidden_states_24, hidden_states_29, hidden_states_36, hidden_states_41, hidden_states_42, pow_11, rsqrt_7, variance_7], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_23 => add_28
# forwarded_states_6 => mul_33
# hidden_states_24 => add_17
# hidden_states_29 => add_20
# hidden_states_36 => add_24
# hidden_states_41 => add_27
# hidden_states_42 => mul_32
# pow_11 => pow_11
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/t4/ct46esogxggodjq4c2lz6ioenief4zh7hif52ygsbpvfwiraytsl.py
# Source Nodes: [add_35, forwarded_states_10, hidden_states_48, hidden_states_53, hidden_states_60, hidden_states_65, hidden_states_66, pow_17, rsqrt_11, variance_11], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_35 => add_42
# forwarded_states_10 => mul_51
# hidden_states_48 => add_31
# hidden_states_53 => add_34
# hidden_states_60 => add_38
# hidden_states_65 => add_41
# hidden_states_66 => mul_50
# pow_17 => pow_17
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp3 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5l4fdte4vdcmesex4qomtxqcd4hobqea6hutcihy46bjiosbjl.py
# Source Nodes: [softmax_9], Original ATen: [aten._softmax]
# softmax_9 => amax_9, div_13, exp_9, sub_14, sum_10
triton_per_fused__softmax_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp11, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4w2mswrpq4wajlgb7rpnfasvyarvbduxtxinmtwsvyncf7lbnp.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24
triton_red_fused__log_softmax_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 62528
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
        tmp0 = tl.load(in_ptr0 + (r1 + (62528*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwwzjycnil37nb6cmx5danbuh7xobhvkjwgodjl45zknd5fdan2.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax_24
triton_per_fused__log_softmax_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwrxygfraw2czevzbwh4ab7qhhxowgwhi5hljlw2q2i6aipqyzd6.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => exp_24, sub_29, sum_25
triton_red_fused__log_softmax_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 62528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 4)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (62528*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl.exp(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqtrezwtflqhf3cb46j3r2smwhkwtzm3vl2gipw3houfrvepivs.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => exp_24, sub_29, sum_25
triton_per_fused__log_softmax_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2o4cc5nco5nz4lpx2vmi2fer5gqsckrmgoqregmvju7hpn3cltp.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type_7, div_28, full_default_7, ne_1, ne_2, neg_1, sum_26, sum_27, where_3
triton_per_fused_nll_loss_forward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tmp4 + 250112
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 250112), "index out of bounds: 0 <= tmp7 < 250112")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (250112*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp12 = tl.log(tmp11)
    tmp13 = tmp10 - tmp12
    tmp14 = -tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp2.to(tl.int64)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp20 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp27, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1 = args
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
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (250112, 512), (512, 1))
    assert_size_stride(arg43_1, (384, 512), (512, 1))
    assert_size_stride(arg44_1, (384, 512), (512, 1))
    assert_size_stride(arg45_1, (384, 512), (512, 1))
    assert_size_stride(arg46_1, (32, 6), (6, 1))
    assert_size_stride(arg47_1, (512, 384), (384, 1))
    assert_size_stride(arg48_1, (1024, 512), (512, 1))
    assert_size_stride(arg49_1, (1024, 512), (512, 1))
    assert_size_stride(arg50_1, (512, 1024), (1024, 1))
    assert_size_stride(arg51_1, (384, 512), (512, 1))
    assert_size_stride(arg52_1, (384, 512), (512, 1))
    assert_size_stride(arg53_1, (384, 512), (512, 1))
    assert_size_stride(arg54_1, (512, 384), (384, 1))
    assert_size_stride(arg55_1, (1024, 512), (512, 1))
    assert_size_stride(arg56_1, (1024, 512), (512, 1))
    assert_size_stride(arg57_1, (512, 1024), (1024, 1))
    assert_size_stride(arg58_1, (384, 512), (512, 1))
    assert_size_stride(arg59_1, (384, 512), (512, 1))
    assert_size_stride(arg60_1, (384, 512), (512, 1))
    assert_size_stride(arg61_1, (512, 384), (384, 1))
    assert_size_stride(arg62_1, (1024, 512), (512, 1))
    assert_size_stride(arg63_1, (1024, 512), (512, 1))
    assert_size_stride(arg64_1, (512, 1024), (1024, 1))
    assert_size_stride(arg65_1, (384, 512), (512, 1))
    assert_size_stride(arg66_1, (384, 512), (512, 1))
    assert_size_stride(arg67_1, (384, 512), (512, 1))
    assert_size_stride(arg68_1, (512, 384), (384, 1))
    assert_size_stride(arg69_1, (1024, 512), (512, 1))
    assert_size_stride(arg70_1, (1024, 512), (512, 1))
    assert_size_stride(arg71_1, (512, 1024), (1024, 1))
    assert_size_stride(arg72_1, (384, 512), (512, 1))
    assert_size_stride(arg73_1, (384, 512), (512, 1))
    assert_size_stride(arg74_1, (384, 512), (512, 1))
    assert_size_stride(arg75_1, (512, 384), (384, 1))
    assert_size_stride(arg76_1, (1024, 512), (512, 1))
    assert_size_stride(arg77_1, (1024, 512), (512, 1))
    assert_size_stride(arg78_1, (512, 1024), (1024, 1))
    assert_size_stride(arg79_1, (384, 512), (512, 1))
    assert_size_stride(arg80_1, (384, 512), (512, 1))
    assert_size_stride(arg81_1, (384, 512), (512, 1))
    assert_size_stride(arg82_1, (512, 384), (384, 1))
    assert_size_stride(arg83_1, (1024, 512), (512, 1))
    assert_size_stride(arg84_1, (1024, 512), (512, 1))
    assert_size_stride(arg85_1, (512, 1024), (1024, 1))
    assert_size_stride(arg86_1, (384, 512), (512, 1))
    assert_size_stride(arg87_1, (384, 512), (512, 1))
    assert_size_stride(arg88_1, (384, 512), (512, 1))
    assert_size_stride(arg89_1, (512, 384), (384, 1))
    assert_size_stride(arg90_1, (1024, 512), (512, 1))
    assert_size_stride(arg91_1, (1024, 512), (512, 1))
    assert_size_stride(arg92_1, (512, 1024), (1024, 1))
    assert_size_stride(arg93_1, (384, 512), (512, 1))
    assert_size_stride(arg94_1, (384, 512), (512, 1))
    assert_size_stride(arg95_1, (384, 512), (512, 1))
    assert_size_stride(arg96_1, (512, 384), (384, 1))
    assert_size_stride(arg97_1, (1024, 512), (512, 1))
    assert_size_stride(arg98_1, (1024, 512), (512, 1))
    assert_size_stride(arg99_1, (512, 1024), (1024, 1))
    assert_size_stride(arg100_1, (384, 512), (512, 1))
    assert_size_stride(arg101_1, (384, 512), (512, 1))
    assert_size_stride(arg102_1, (384, 512), (512, 1))
    assert_size_stride(arg103_1, (32, 6), (6, 1))
    assert_size_stride(arg104_1, (512, 384), (384, 1))
    assert_size_stride(arg105_1, (384, 512), (512, 1))
    assert_size_stride(arg106_1, (384, 512), (512, 1))
    assert_size_stride(arg107_1, (384, 512), (512, 1))
    assert_size_stride(arg108_1, (512, 384), (384, 1))
    assert_size_stride(arg109_1, (1024, 512), (512, 1))
    assert_size_stride(arg110_1, (1024, 512), (512, 1))
    assert_size_stride(arg111_1, (512, 1024), (1024, 1))
    assert_size_stride(arg112_1, (384, 512), (512, 1))
    assert_size_stride(arg113_1, (384, 512), (512, 1))
    assert_size_stride(arg114_1, (384, 512), (512, 1))
    assert_size_stride(arg115_1, (512, 384), (384, 1))
    assert_size_stride(arg116_1, (384, 512), (512, 1))
    assert_size_stride(arg117_1, (384, 512), (512, 1))
    assert_size_stride(arg118_1, (384, 512), (512, 1))
    assert_size_stride(arg119_1, (512, 384), (384, 1))
    assert_size_stride(arg120_1, (1024, 512), (512, 1))
    assert_size_stride(arg121_1, (1024, 512), (512, 1))
    assert_size_stride(arg122_1, (512, 1024), (1024, 1))
    assert_size_stride(arg123_1, (384, 512), (512, 1))
    assert_size_stride(arg124_1, (384, 512), (512, 1))
    assert_size_stride(arg125_1, (384, 512), (512, 1))
    assert_size_stride(arg126_1, (512, 384), (384, 1))
    assert_size_stride(arg127_1, (384, 512), (512, 1))
    assert_size_stride(arg128_1, (384, 512), (512, 1))
    assert_size_stride(arg129_1, (384, 512), (512, 1))
    assert_size_stride(arg130_1, (512, 384), (384, 1))
    assert_size_stride(arg131_1, (1024, 512), (512, 1))
    assert_size_stride(arg132_1, (1024, 512), (512, 1))
    assert_size_stride(arg133_1, (512, 1024), (1024, 1))
    assert_size_stride(arg134_1, (384, 512), (512, 1))
    assert_size_stride(arg135_1, (384, 512), (512, 1))
    assert_size_stride(arg136_1, (384, 512), (512, 1))
    assert_size_stride(arg137_1, (512, 384), (384, 1))
    assert_size_stride(arg138_1, (384, 512), (512, 1))
    assert_size_stride(arg139_1, (384, 512), (512, 1))
    assert_size_stride(arg140_1, (384, 512), (512, 1))
    assert_size_stride(arg141_1, (512, 384), (384, 1))
    assert_size_stride(arg142_1, (1024, 512), (512, 1))
    assert_size_stride(arg143_1, (1024, 512), (512, 1))
    assert_size_stride(arg144_1, (512, 1024), (1024, 1))
    assert_size_stride(arg145_1, (384, 512), (512, 1))
    assert_size_stride(arg146_1, (384, 512), (512, 1))
    assert_size_stride(arg147_1, (384, 512), (512, 1))
    assert_size_stride(arg148_1, (512, 384), (384, 1))
    assert_size_stride(arg149_1, (384, 512), (512, 1))
    assert_size_stride(arg150_1, (384, 512), (512, 1))
    assert_size_stride(arg151_1, (384, 512), (512, 1))
    assert_size_stride(arg152_1, (512, 384), (384, 1))
    assert_size_stride(arg153_1, (1024, 512), (512, 1))
    assert_size_stride(arg154_1, (1024, 512), (512, 1))
    assert_size_stride(arg155_1, (512, 1024), (1024, 1))
    assert_size_stride(arg156_1, (384, 512), (512, 1))
    assert_size_stride(arg157_1, (384, 512), (512, 1))
    assert_size_stride(arg158_1, (384, 512), (512, 1))
    assert_size_stride(arg159_1, (512, 384), (384, 1))
    assert_size_stride(arg160_1, (384, 512), (512, 1))
    assert_size_stride(arg161_1, (384, 512), (512, 1))
    assert_size_stride(arg162_1, (384, 512), (512, 1))
    assert_size_stride(arg163_1, (512, 384), (384, 1))
    assert_size_stride(arg164_1, (1024, 512), (512, 1))
    assert_size_stride(arg165_1, (1024, 512), (512, 1))
    assert_size_stride(arg166_1, (512, 1024), (1024, 1))
    assert_size_stride(arg167_1, (384, 512), (512, 1))
    assert_size_stride(arg168_1, (384, 512), (512, 1))
    assert_size_stride(arg169_1, (384, 512), (512, 1))
    assert_size_stride(arg170_1, (512, 384), (384, 1))
    assert_size_stride(arg171_1, (384, 512), (512, 1))
    assert_size_stride(arg172_1, (384, 512), (512, 1))
    assert_size_stride(arg173_1, (384, 512), (512, 1))
    assert_size_stride(arg174_1, (512, 384), (384, 1))
    assert_size_stride(arg175_1, (1024, 512), (512, 1))
    assert_size_stride(arg176_1, (1024, 512), (512, 1))
    assert_size_stride(arg177_1, (512, 1024), (1024, 1))
    assert_size_stride(arg178_1, (384, 512), (512, 1))
    assert_size_stride(arg179_1, (384, 512), (512, 1))
    assert_size_stride(arg180_1, (384, 512), (512, 1))
    assert_size_stride(arg181_1, (512, 384), (384, 1))
    assert_size_stride(arg182_1, (384, 512), (512, 1))
    assert_size_stride(arg183_1, (384, 512), (512, 1))
    assert_size_stride(arg184_1, (384, 512), (512, 1))
    assert_size_stride(arg185_1, (512, 384), (384, 1))
    assert_size_stride(arg186_1, (1024, 512), (512, 1))
    assert_size_stride(arg187_1, (1024, 512), (512, 1))
    assert_size_stride(arg188_1, (512, 1024), (1024, 1))
    assert_size_stride(arg189_1, (250112, 512), (512, 1))
    assert_size_stride(arg190_1, (1, 128), (128, 1))
    assert_size_stride(arg191_1, (1, 128), (128, 1))
    assert_size_stride(arg192_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_52, hidden_states_102, inputs_embeds_1, normed_hidden_states_8, pow_26, rsqrt_17, variance_17], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0.run(arg192_1, arg42_1, arg17_1, buf1, 128, 512, grid=grid(128), stream=stream0)
        del arg17_1
        buf2 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (128, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 384), (1, 512), 0), out=buf2)
        del arg100_1
        buf3 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (128, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 384), (1, 512), 0), out=buf3)
        del arg101_1
        buf4 = empty((6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf3, (6, 64, 128), (64, 1, 384), 0), out=buf4)
        buf9 = empty((1, 6, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [softmax_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_1.run(buf4, arg103_1, buf9, 768, 128, grid=grid(768), stream=stream0)
        buf8 = buf2; del buf2  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (128, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 384), (1, 512), 0), out=buf8)
        del arg102_1
        buf10 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf8, (6, 128, 64), (64, 384, 1), 0), out=buf10)
        buf11 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf10, buf11, 49152, grid=grid(49152), stream=stream0)
        buf12 = reinterpret_tensor(buf1, (128, 512), (512, 1), 0); del buf1  # reuse
        # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (128, 384), (384, 1), 0), reinterpret_tensor(arg104_1, (384, 512), (1, 384), 0), out=buf12)
        del arg104_1
        buf14 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf17 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_57, hidden_states_1, hidden_states_106, hidden_states_107, inputs_embeds, inputs_embeds_1, normed_hidden_states, normed_hidden_states_9, pow_1, pow_27, rsqrt, rsqrt_18, variance, variance_18], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_3.run(arg192_1, arg42_1, buf12, arg190_1, arg18_1, arg0_1, buf14, buf17, 128, 512, grid=grid(128), stream=stream0)
        del arg0_1
        del arg18_1
        buf15 = reinterpret_tensor(buf11, (128, 384), (384, 1), 0); del buf11  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (128, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 384), (1, 512), 0), out=buf15)
        del arg105_1
        buf18 = reinterpret_tensor(buf10, (128, 384), (384, 1), 0); del buf10  # reuse
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (128, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 384), (1, 512), 0), out=buf18)
        del arg43_1
        buf19 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (128, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 384), (1, 512), 0), out=buf19)
        del arg44_1
        buf20 = reinterpret_tensor(buf9, (6, 128, 128), (16384, 128, 1), 0); del buf9  # reuse
        # Source Nodes: [scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf19, (6, 64, 128), (64, 1, 384), 0), out=buf20)
        buf24 = reinterpret_tensor(buf4, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf4  # reuse
        # Source Nodes: [softmax], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf20, arg46_1, buf24, 768, 128, grid=grid(768), stream=stream0)
        buf23 = buf19; del buf19  # reuse
        # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (128, 512), (512, 1), 0), reinterpret_tensor(arg45_1, (512, 384), (1, 512), 0), out=buf23)
        del arg45_1
        buf25 = reinterpret_tensor(buf18, (6, 128, 64), (8192, 64, 1), 0); del buf18  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf23, (6, 128, 64), (64, 384, 1), 0), out=buf25)
        buf26 = reinterpret_tensor(buf23, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf23  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf25, buf26, 49152, grid=grid(49152), stream=stream0)
        buf27 = reinterpret_tensor(buf17, (128, 512), (512, 1), 0); del buf17  # reuse
        # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (128, 384), (384, 1), 0), reinterpret_tensor(arg47_1, (384, 512), (1, 384), 0), out=buf27)
        del arg47_1
        buf29 = buf14; del buf14  # reuse
        # Source Nodes: [add_5, forwarded_states, hidden_states_5, hidden_states_6, inputs_embeds, pow_2, rsqrt_1, variance_1], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5.run(arg190_1, arg42_1, buf27, arg1_1, buf29, 128, 512, grid=grid(128), stream=stream0)
        del arg1_1
        buf30 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (128, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 1024), (1, 512), 0), out=buf30)
        del arg48_1
        buf31 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (128, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 1024), (1, 512), 0), out=buf31)
        del arg49_1
        buf32 = reinterpret_tensor(buf30, (1, 128, 1024), (131072, 1024, 1), 0); del buf30  # reuse
        # Source Nodes: [add_6, add_7, hidden_gelu, hidden_states_7, mul_7, mul_8, mul_9, pow_3, tanh], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf32, buf31, 131072, grid=grid(131072), stream=stream0)
        buf33 = reinterpret_tensor(buf29, (128, 512), (512, 1), 0); del buf29  # reuse
        # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg50_1, (1024, 512), (1, 1024), 0), out=buf33)
        del arg50_1
        buf35 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_9, hidden_states_12, hidden_states_13, hidden_states_5, inputs_embeds, normed_hidden_states_1, pow_4, rsqrt_2, variance_2], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7.run(arg190_1, arg42_1, buf27, buf33, arg2_1, buf35, 128, 512, grid=grid(128), stream=stream0)
        del arg2_1
        buf36 = reinterpret_tensor(buf26, (128, 384), (384, 1), 0); del buf26  # reuse
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (128, 512), (512, 1), 0), reinterpret_tensor(arg51_1, (512, 384), (1, 512), 0), out=buf36)
        del arg51_1
        buf37 = reinterpret_tensor(buf25, (128, 384), (384, 1), 0); del buf25  # reuse
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (128, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 384), (1, 512), 0), out=buf37)
        del arg52_1
        buf38 = reinterpret_tensor(buf24, (6, 128, 128), (16384, 128, 1), 0); del buf24  # reuse
        # Source Nodes: [scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf37, (6, 64, 128), (64, 1, 384), 0), out=buf38)
        buf42 = reinterpret_tensor(buf20, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf20  # reuse
        # Source Nodes: [softmax_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf38, arg46_1, buf42, 768, 128, grid=grid(768), stream=stream0)
        buf41 = buf37; del buf37  # reuse
        # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (128, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 384), (1, 512), 0), out=buf41)
        del arg53_1
        buf43 = reinterpret_tensor(buf36, (6, 128, 64), (8192, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf41, (6, 128, 64), (64, 384, 1), 0), out=buf43)
        buf44 = reinterpret_tensor(buf41, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf41  # reuse
        # Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf43, buf44, 49152, grid=grid(49152), stream=stream0)
        buf45 = reinterpret_tensor(buf35, (128, 512), (512, 1), 0); del buf35  # reuse
        # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (128, 384), (384, 1), 0), reinterpret_tensor(arg54_1, (384, 512), (1, 384), 0), out=buf45)
        del arg54_1
        buf46 = reinterpret_tensor(buf27, (1, 128, 512), (65536, 512, 1), 0); del buf27  # reuse
        buf48 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_11, forwarded_states_2, hidden_states_12, hidden_states_17, hidden_states_18, hidden_states_5, inputs_embeds, pow_5, rsqrt_3, variance_3], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8.run(buf46, arg190_1, arg42_1, buf33, buf45, arg3_1, buf48, 128, 512, grid=grid(128), stream=stream0)
        del arg190_1
        del arg3_1
        buf49 = reinterpret_tensor(buf32, (128, 1024), (1024, 1), 0); del buf32  # reuse
        # Source Nodes: [l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (128, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 1024), (1, 512), 0), out=buf49)
        del arg55_1
        buf50 = buf31; del buf31  # reuse
        # Source Nodes: [hidden_linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (128, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 1024), (1, 512), 0), out=buf50)
        del arg56_1
        buf51 = reinterpret_tensor(buf49, (1, 128, 1024), (131072, 1024, 1), 0); del buf49  # reuse
        # Source Nodes: [add_12, add_13, hidden_gelu_1, hidden_states_19, mul_16, mul_17, mul_18, pow_6, tanh_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf51, buf50, 131072, grid=grid(131072), stream=stream0)
        buf52 = reinterpret_tensor(buf48, (128, 512), (512, 1), 0); del buf48  # reuse
        # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 512), (1, 1024), 0), out=buf52)
        del arg57_1
        buf54 = reinterpret_tensor(buf45, (1, 128, 512), (65536, 512, 1), 0); del buf45  # reuse
        # Source Nodes: [add_15, hidden_states_24, hidden_states_25, normed_hidden_states_2, pow_7, rsqrt_4, variance_4], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf46, buf52, arg4_1, buf54, 128, 512, grid=grid(128), stream=stream0)
        del arg4_1
        buf55 = reinterpret_tensor(buf44, (128, 384), (384, 1), 0); del buf44  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (128, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 384), (1, 512), 0), out=buf55)
        del arg58_1
        buf56 = reinterpret_tensor(buf43, (128, 384), (384, 1), 0); del buf43  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (128, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 384), (1, 512), 0), out=buf56)
        del arg59_1
        buf57 = reinterpret_tensor(buf42, (6, 128, 128), (16384, 128, 1), 0); del buf42  # reuse
        # Source Nodes: [scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf56, (6, 64, 128), (64, 1, 384), 0), out=buf57)
        buf61 = reinterpret_tensor(buf38, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf38  # reuse
        # Source Nodes: [softmax_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf57, arg46_1, buf61, 768, 128, grid=grid(768), stream=stream0)
        buf60 = buf56; del buf56  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (128, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 384), (1, 512), 0), out=buf60)
        del arg60_1
        buf62 = reinterpret_tensor(buf55, (6, 128, 64), (8192, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf61, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf60, (6, 128, 64), (64, 384, 1), 0), out=buf62)
        buf63 = reinterpret_tensor(buf60, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf60  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf62, buf63, 49152, grid=grid(49152), stream=stream0)
        buf64 = reinterpret_tensor(buf54, (128, 512), (512, 1), 0); del buf54  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (128, 384), (384, 1), 0), reinterpret_tensor(arg61_1, (384, 512), (1, 384), 0), out=buf64)
        del arg61_1
        buf66 = reinterpret_tensor(buf33, (1, 128, 512), (65536, 512, 1), 0); del buf33  # reuse
        # Source Nodes: [add_17, forwarded_states_4, hidden_states_24, hidden_states_29, hidden_states_30, pow_8, rsqrt_5, variance_5], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf46, buf52, buf64, arg5_1, buf66, 128, 512, grid=grid(128), stream=stream0)
        del arg5_1
        buf67 = reinterpret_tensor(buf51, (128, 1024), (1024, 1), 0); del buf51  # reuse
        # Source Nodes: [l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (128, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 1024), (1, 512), 0), out=buf67)
        del arg62_1
        buf68 = buf50; del buf50  # reuse
        # Source Nodes: [hidden_linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (128, 512), (512, 1), 0), reinterpret_tensor(arg63_1, (512, 1024), (1, 512), 0), out=buf68)
        del arg63_1
        buf69 = reinterpret_tensor(buf67, (1, 128, 1024), (131072, 1024, 1), 0); del buf67  # reuse
        # Source Nodes: [add_18, add_19, hidden_gelu_2, hidden_states_31, mul_25, mul_26, mul_27, pow_9, tanh_2], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf69, buf68, 131072, grid=grid(131072), stream=stream0)
        buf70 = reinterpret_tensor(buf66, (128, 512), (512, 1), 0); del buf66  # reuse
        # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 512), (1, 1024), 0), out=buf70)
        del arg64_1
        buf72 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, hidden_states_24, hidden_states_29, hidden_states_36, hidden_states_37, normed_hidden_states_3, pow_10, rsqrt_6, variance_6], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf46, buf52, buf64, buf70, arg6_1, buf72, 128, 512, grid=grid(128), stream=stream0)
        del arg6_1
        buf73 = reinterpret_tensor(buf63, (128, 384), (384, 1), 0); del buf63  # reuse
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (128, 512), (512, 1), 0), reinterpret_tensor(arg65_1, (512, 384), (1, 512), 0), out=buf73)
        del arg65_1
        buf74 = reinterpret_tensor(buf62, (128, 384), (384, 1), 0); del buf62  # reuse
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (128, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 384), (1, 512), 0), out=buf74)
        del arg66_1
        buf75 = reinterpret_tensor(buf61, (6, 128, 128), (16384, 128, 1), 0); del buf61  # reuse
        # Source Nodes: [scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf74, (6, 64, 128), (64, 1, 384), 0), out=buf75)
        buf79 = reinterpret_tensor(buf57, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf57  # reuse
        # Source Nodes: [softmax_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf75, arg46_1, buf79, 768, 128, grid=grid(768), stream=stream0)
        buf78 = buf74; del buf74  # reuse
        # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (128, 512), (512, 1), 0), reinterpret_tensor(arg67_1, (512, 384), (1, 512), 0), out=buf78)
        del arg67_1
        buf80 = reinterpret_tensor(buf73, (6, 128, 64), (8192, 64, 1), 0); del buf73  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf78, (6, 128, 64), (64, 384, 1), 0), out=buf80)
        buf81 = reinterpret_tensor(buf78, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf80, buf81, 49152, grid=grid(49152), stream=stream0)
        buf82 = reinterpret_tensor(buf72, (128, 512), (512, 1), 0); del buf72  # reuse
        # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (128, 384), (384, 1), 0), reinterpret_tensor(arg68_1, (384, 512), (1, 384), 0), out=buf82)
        del arg68_1
        buf83 = buf46; del buf46  # reuse
        buf85 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_23, forwarded_states_6, hidden_states_24, hidden_states_29, hidden_states_36, hidden_states_41, hidden_states_42, pow_11, rsqrt_7, variance_7], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf83, buf52, buf64, buf70, buf82, arg7_1, buf85, 128, 512, grid=grid(128), stream=stream0)
        del arg7_1
        buf86 = reinterpret_tensor(buf69, (128, 1024), (1024, 1), 0); del buf69  # reuse
        # Source Nodes: [l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (128, 512), (512, 1), 0), reinterpret_tensor(arg69_1, (512, 1024), (1, 512), 0), out=buf86)
        del arg69_1
        buf87 = buf68; del buf68  # reuse
        # Source Nodes: [hidden_linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (128, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 1024), (1, 512), 0), out=buf87)
        del arg70_1
        buf88 = reinterpret_tensor(buf86, (1, 128, 1024), (131072, 1024, 1), 0); del buf86  # reuse
        # Source Nodes: [add_24, add_25, hidden_gelu_3, hidden_states_43, mul_34, mul_35, mul_36, pow_12, tanh_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf88, buf87, 131072, grid=grid(131072), stream=stream0)
        buf89 = reinterpret_tensor(buf85, (128, 512), (512, 1), 0); del buf85  # reuse
        # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 512), (1, 1024), 0), out=buf89)
        del arg71_1
        buf91 = reinterpret_tensor(buf82, (1, 128, 512), (65536, 512, 1), 0); del buf82  # reuse
        # Source Nodes: [add_27, hidden_states_48, hidden_states_49, normed_hidden_states_4, pow_13, rsqrt_8, variance_8], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf83, buf89, arg8_1, buf91, 128, 512, grid=grid(128), stream=stream0)
        del arg8_1
        buf92 = reinterpret_tensor(buf81, (128, 384), (384, 1), 0); del buf81  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 384), (1, 512), 0), out=buf92)
        del arg72_1
        buf93 = reinterpret_tensor(buf80, (128, 384), (384, 1), 0); del buf80  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 512), (512, 1), 0), reinterpret_tensor(arg73_1, (512, 384), (1, 512), 0), out=buf93)
        del arg73_1
        buf94 = reinterpret_tensor(buf79, (6, 128, 128), (16384, 128, 1), 0); del buf79  # reuse
        # Source Nodes: [scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf92, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf93, (6, 64, 128), (64, 1, 384), 0), out=buf94)
        buf98 = reinterpret_tensor(buf75, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf75  # reuse
        # Source Nodes: [softmax_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf94, arg46_1, buf98, 768, 128, grid=grid(768), stream=stream0)
        buf97 = buf93; del buf93  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 384), (1, 512), 0), out=buf97)
        del arg74_1
        buf99 = reinterpret_tensor(buf92, (6, 128, 64), (8192, 64, 1), 0); del buf92  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf97, (6, 128, 64), (64, 384, 1), 0), out=buf99)
        buf100 = reinterpret_tensor(buf97, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf99, buf100, 49152, grid=grid(49152), stream=stream0)
        buf101 = reinterpret_tensor(buf91, (128, 512), (512, 1), 0); del buf91  # reuse
        # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (128, 384), (384, 1), 0), reinterpret_tensor(arg75_1, (384, 512), (1, 384), 0), out=buf101)
        del arg75_1
        buf103 = reinterpret_tensor(buf70, (1, 128, 512), (65536, 512, 1), 0); del buf70  # reuse
        # Source Nodes: [add_29, forwarded_states_8, hidden_states_48, hidden_states_53, hidden_states_54, pow_14, rsqrt_9, variance_9], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf83, buf89, buf101, arg9_1, buf103, 128, 512, grid=grid(128), stream=stream0)
        del arg9_1
        buf104 = reinterpret_tensor(buf88, (128, 1024), (1024, 1), 0); del buf88  # reuse
        # Source Nodes: [l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 1024), (1, 512), 0), out=buf104)
        del arg76_1
        buf105 = buf87; del buf87  # reuse
        # Source Nodes: [hidden_linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 1024), (1, 512), 0), out=buf105)
        del arg77_1
        buf106 = reinterpret_tensor(buf104, (1, 128, 1024), (131072, 1024, 1), 0); del buf104  # reuse
        # Source Nodes: [add_30, add_31, hidden_gelu_4, hidden_states_55, mul_43, mul_44, mul_45, pow_15, tanh_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf106, buf105, 131072, grid=grid(131072), stream=stream0)
        buf107 = reinterpret_tensor(buf103, (128, 512), (512, 1), 0); del buf103  # reuse
        # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg78_1, (1024, 512), (1, 1024), 0), out=buf107)
        del arg78_1
        buf109 = reinterpret_tensor(buf64, (1, 128, 512), (65536, 512, 1), 0); del buf64  # reuse
        # Source Nodes: [add_33, hidden_states_48, hidden_states_53, hidden_states_60, hidden_states_61, normed_hidden_states_5, pow_16, rsqrt_10, variance_10], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf83, buf89, buf101, buf107, arg10_1, buf109, 128, 512, grid=grid(128), stream=stream0)
        del arg10_1
        buf110 = reinterpret_tensor(buf100, (128, 384), (384, 1), 0); del buf100  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (128, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 384), (1, 512), 0), out=buf110)
        del arg79_1
        buf111 = reinterpret_tensor(buf99, (128, 384), (384, 1), 0); del buf99  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (128, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 384), (1, 512), 0), out=buf111)
        del arg80_1
        buf112 = reinterpret_tensor(buf98, (6, 128, 128), (16384, 128, 1), 0); del buf98  # reuse
        # Source Nodes: [scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf111, (6, 64, 128), (64, 1, 384), 0), out=buf112)
        buf116 = reinterpret_tensor(buf94, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf94  # reuse
        # Source Nodes: [softmax_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf112, arg46_1, buf116, 768, 128, grid=grid(768), stream=stream0)
        buf115 = buf111; del buf111  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (128, 512), (512, 1), 0), reinterpret_tensor(arg81_1, (512, 384), (1, 512), 0), out=buf115)
        del arg81_1
        buf117 = reinterpret_tensor(buf110, (6, 128, 64), (8192, 64, 1), 0); del buf110  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf115, (6, 128, 64), (64, 384, 1), 0), out=buf117)
        buf118 = reinterpret_tensor(buf115, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf115  # reuse
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf117, buf118, 49152, grid=grid(49152), stream=stream0)
        buf119 = reinterpret_tensor(buf109, (128, 512), (512, 1), 0); del buf109  # reuse
        # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (128, 384), (384, 1), 0), reinterpret_tensor(arg82_1, (384, 512), (1, 384), 0), out=buf119)
        del arg82_1
        buf120 = reinterpret_tensor(buf101, (1, 128, 512), (65536, 512, 1), 0); del buf101  # reuse
        buf122 = reinterpret_tensor(buf52, (1, 128, 512), (65536, 512, 1), 0); del buf52  # reuse
        # Source Nodes: [add_35, forwarded_states_10, hidden_states_48, hidden_states_53, hidden_states_60, hidden_states_65, hidden_states_66, pow_17, rsqrt_11, variance_11], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf120, buf83, buf89, buf107, buf119, arg11_1, buf122, 128, 512, grid=grid(128), stream=stream0)
        del arg11_1
        buf123 = reinterpret_tensor(buf106, (128, 1024), (1024, 1), 0); del buf106  # reuse
        # Source Nodes: [l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (128, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 1024), (1, 512), 0), out=buf123)
        del arg83_1
        buf124 = buf105; del buf105  # reuse
        # Source Nodes: [hidden_linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (128, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 1024), (1, 512), 0), out=buf124)
        del arg84_1
        buf125 = reinterpret_tensor(buf123, (1, 128, 1024), (131072, 1024, 1), 0); del buf123  # reuse
        # Source Nodes: [add_36, add_37, hidden_gelu_5, hidden_states_67, mul_52, mul_53, mul_54, pow_18, tanh_5], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf125, buf124, 131072, grid=grid(131072), stream=stream0)
        buf126 = reinterpret_tensor(buf122, (128, 512), (512, 1), 0); del buf122  # reuse
        # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 512), (1, 1024), 0), out=buf126)
        del arg85_1
        buf128 = reinterpret_tensor(buf89, (1, 128, 512), (65536, 512, 1), 0); del buf89  # reuse
        # Source Nodes: [add_39, hidden_states_72, hidden_states_73, normed_hidden_states_6, pow_19, rsqrt_12, variance_12], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf120, buf126, arg12_1, buf128, 128, 512, grid=grid(128), stream=stream0)
        del arg12_1
        buf129 = reinterpret_tensor(buf118, (128, 384), (384, 1), 0); del buf118  # reuse
        # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 384), (1, 512), 0), out=buf129)
        del arg86_1
        buf130 = reinterpret_tensor(buf117, (128, 384), (384, 1), 0); del buf117  # reuse
        # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 384), (1, 512), 0), out=buf130)
        del arg87_1
        buf131 = reinterpret_tensor(buf116, (6, 128, 128), (16384, 128, 1), 0); del buf116  # reuse
        # Source Nodes: [scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf130, (6, 64, 128), (64, 1, 384), 0), out=buf131)
        buf135 = reinterpret_tensor(buf112, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf112  # reuse
        # Source Nodes: [softmax_6], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf131, arg46_1, buf135, 768, 128, grid=grid(768), stream=stream0)
        buf134 = buf130; del buf130  # reuse
        # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (128, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 384), (1, 512), 0), out=buf134)
        del arg88_1
        buf136 = reinterpret_tensor(buf129, (6, 128, 64), (8192, 64, 1), 0); del buf129  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf134, (6, 128, 64), (64, 384, 1), 0), out=buf136)
        buf137 = reinterpret_tensor(buf134, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf134  # reuse
        # Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf136, buf137, 49152, grid=grid(49152), stream=stream0)
        buf138 = reinterpret_tensor(buf128, (128, 512), (512, 1), 0); del buf128  # reuse
        # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (128, 384), (384, 1), 0), reinterpret_tensor(arg89_1, (384, 512), (1, 384), 0), out=buf138)
        del arg89_1
        buf140 = buf83; del buf83  # reuse
        # Source Nodes: [add_41, forwarded_states_12, hidden_states_72, hidden_states_77, hidden_states_78, pow_20, rsqrt_13, variance_13], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf120, buf126, buf138, arg13_1, buf140, 128, 512, grid=grid(128), stream=stream0)
        del arg13_1
        buf141 = reinterpret_tensor(buf125, (128, 1024), (1024, 1), 0); del buf125  # reuse
        # Source Nodes: [l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (128, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 1024), (1, 512), 0), out=buf141)
        del arg90_1
        buf142 = buf124; del buf124  # reuse
        # Source Nodes: [hidden_linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (128, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 1024), (1, 512), 0), out=buf142)
        del arg91_1
        buf143 = reinterpret_tensor(buf141, (1, 128, 1024), (131072, 1024, 1), 0); del buf141  # reuse
        # Source Nodes: [add_42, add_43, hidden_gelu_6, hidden_states_79, mul_61, mul_62, mul_63, pow_21, tanh_6], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf143, buf142, 131072, grid=grid(131072), stream=stream0)
        buf144 = reinterpret_tensor(buf140, (128, 512), (512, 1), 0); del buf140  # reuse
        # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 512), (1, 1024), 0), out=buf144)
        del arg92_1
        buf146 = reinterpret_tensor(buf119, (1, 128, 512), (65536, 512, 1), 0); del buf119  # reuse
        # Source Nodes: [add_45, hidden_states_72, hidden_states_77, hidden_states_84, hidden_states_85, normed_hidden_states_7, pow_22, rsqrt_14, variance_14], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf120, buf126, buf138, buf144, arg14_1, buf146, 128, 512, grid=grid(128), stream=stream0)
        del arg14_1
        buf147 = reinterpret_tensor(buf137, (128, 384), (384, 1), 0); del buf137  # reuse
        # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (128, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 384), (1, 512), 0), out=buf147)
        del arg93_1
        buf148 = reinterpret_tensor(buf136, (128, 384), (384, 1), 0); del buf136  # reuse
        # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (128, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 384), (1, 512), 0), out=buf148)
        del arg94_1
        buf149 = reinterpret_tensor(buf135, (6, 128, 128), (16384, 128, 1), 0); del buf135  # reuse
        # Source Nodes: [scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf147, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf148, (6, 64, 128), (64, 1, 384), 0), out=buf149)
        buf153 = reinterpret_tensor(buf131, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf131  # reuse
        # Source Nodes: [softmax_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf149, arg46_1, buf153, 768, 128, grid=grid(768), stream=stream0)
        del arg46_1
        buf152 = buf148; del buf148  # reuse
        # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (128, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 384), (1, 512), 0), out=buf152)
        del arg95_1
        buf154 = reinterpret_tensor(buf147, (6, 128, 64), (8192, 64, 1), 0); del buf147  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf152, (6, 128, 64), (64, 384, 1), 0), out=buf154)
        buf155 = reinterpret_tensor(buf152, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf152  # reuse
        # Source Nodes: [contiguous_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf154, buf155, 49152, grid=grid(49152), stream=stream0)
        buf156 = reinterpret_tensor(buf146, (128, 512), (512, 1), 0); del buf146  # reuse
        # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (128, 384), (384, 1), 0), reinterpret_tensor(arg96_1, (384, 512), (1, 384), 0), out=buf156)
        del arg96_1
        buf157 = buf120; del buf120  # reuse
        buf159 = reinterpret_tensor(buf107, (1, 128, 512), (65536, 512, 1), 0); del buf107  # reuse
        # Source Nodes: [add_47, forwarded_states_14, hidden_states_72, hidden_states_77, hidden_states_84, hidden_states_89, hidden_states_90, pow_23, rsqrt_15, variance_15], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf157, buf126, buf138, buf144, buf156, arg15_1, buf159, 128, 512, grid=grid(128), stream=stream0)
        del arg15_1
        buf160 = reinterpret_tensor(buf143, (128, 1024), (1024, 1), 0); del buf143  # reuse
        # Source Nodes: [l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (128, 512), (512, 1), 0), reinterpret_tensor(arg97_1, (512, 1024), (1, 512), 0), out=buf160)
        del arg97_1
        buf161 = buf142; del buf142  # reuse
        # Source Nodes: [hidden_linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (128, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 1024), (1, 512), 0), out=buf161)
        del arg98_1
        buf162 = reinterpret_tensor(buf160, (1, 128, 1024), (131072, 1024, 1), 0); del buf160  # reuse
        # Source Nodes: [add_48, add_49, hidden_gelu_7, hidden_states_91, mul_70, mul_71, mul_72, pow_24, tanh_7], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf162, buf161, 131072, grid=grid(131072), stream=stream0)
        buf163 = reinterpret_tensor(buf159, (128, 512), (512, 1), 0); del buf159  # reuse
        # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg99_1, (1024, 512), (1, 1024), 0), out=buf163)
        del arg99_1
        buf165 = reinterpret_tensor(buf156, (1, 128, 512), (65536, 512, 1), 0); del buf156  # reuse
        # Source Nodes: [add_51, hidden_states_96, hidden_states_97, hidden_states_98, pow_25, rsqrt_16, variance_16], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf157, buf163, arg16_1, buf165, 128, 512, grid=grid(128), stream=stream0)
        del arg16_1
        buf166 = reinterpret_tensor(buf155, (128, 384), (384, 1), 0); del buf155  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 384), (1, 512), 0), out=buf166)
        del arg106_1
        buf167 = reinterpret_tensor(buf153, (6, 128, 128), (16384, 128, 1), 0); del buf153  # reuse
        # Source Nodes: [scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf166, (6, 64, 128), (64, 1, 384), 0), out=buf167)
        buf171 = reinterpret_tensor(buf149, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf149  # reuse
        # Source Nodes: [softmax_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf167, buf171, 768, 128, grid=grid(768), stream=stream0)
        buf170 = buf15; del buf15  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 384), (1, 512), 0), out=buf170)
        del arg107_1
        buf172 = buf154; del buf154  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf170, (6, 128, 64), (64, 384, 1), 0), out=buf172)
        buf173 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf172, buf173, 49152, grid=grid(49152), stream=stream0)
        buf174 = buf163; del buf163  # reuse
        # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (128, 384), (384, 1), 0), reinterpret_tensor(arg108_1, (384, 512), (1, 384), 0), out=buf174)
        del arg108_1
        buf176 = buf157; del buf157  # reuse
        # Source Nodes: [add_60, forwarded_states_16, hidden_states_106, hidden_states_110, hidden_states_111, inputs_embeds_1, pow_28, rsqrt_19, variance_19], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7.run(arg192_1, arg42_1, buf12, buf174, arg19_1, buf176, 128, 512, grid=grid(128), stream=stream0)
        del arg19_1
        buf177 = reinterpret_tensor(buf162, (128, 1024), (1024, 1), 0); del buf162  # reuse
        # Source Nodes: [l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (128, 512), (512, 1), 0), reinterpret_tensor(arg109_1, (512, 1024), (1, 512), 0), out=buf177)
        del arg109_1
        buf178 = buf161; del buf161  # reuse
        # Source Nodes: [hidden_linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (128, 512), (512, 1), 0), reinterpret_tensor(arg110_1, (512, 1024), (1, 512), 0), out=buf178)
        del arg110_1
        buf179 = reinterpret_tensor(buf177, (1, 128, 1024), (131072, 1024, 1), 0); del buf177  # reuse
        # Source Nodes: [add_61, add_62, hidden_gelu_8, hidden_states_112, mul_87, mul_88, mul_89, pow_29, tanh_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf179, buf178, 131072, grid=grid(131072), stream=stream0)
        buf180 = reinterpret_tensor(buf176, (128, 512), (512, 1), 0); del buf176  # reuse
        # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg111_1, (1024, 512), (1, 1024), 0), out=buf180)
        del arg111_1
        buf181 = reinterpret_tensor(buf12, (1, 128, 512), (65536, 512, 1), 0); del buf12  # reuse
        buf183 = reinterpret_tensor(buf144, (1, 128, 512), (65536, 512, 1), 0); del buf144  # reuse
        # Source Nodes: [add_64, hidden_states_106, hidden_states_110, hidden_states_117, hidden_states_118, inputs_embeds_1, normed_hidden_states_10, pow_30, rsqrt_20, variance_20], Original ATen: [aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8.run(buf181, arg192_1, arg42_1, buf174, buf180, arg20_1, buf183, 128, 512, grid=grid(128), stream=stream0)
        del arg192_1
        del arg20_1
        del arg42_1
        buf184 = reinterpret_tensor(buf173, (128, 384), (384, 1), 0); del buf173  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (128, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 384), (1, 512), 0), out=buf184)
        del arg112_1
        buf185 = reinterpret_tensor(buf172, (128, 384), (384, 1), 0); del buf172  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (128, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 384), (1, 512), 0), out=buf185)
        del arg113_1
        buf186 = reinterpret_tensor(buf171, (6, 128, 128), (16384, 128, 1), 0); del buf171  # reuse
        # Source Nodes: [scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf185, (6, 64, 128), (64, 1, 384), 0), out=buf186)
        buf191 = reinterpret_tensor(buf167, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf167  # reuse
        # Source Nodes: [softmax_10], Original ATen: [aten._softmax]
        triton_per_fused__softmax_1.run(buf186, arg103_1, buf191, 768, 128, grid=grid(768), stream=stream0)
        buf190 = buf184; del buf184  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (128, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 384), (1, 512), 0), out=buf190)
        del arg114_1
        buf192 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf190, (6, 128, 64), (64, 384, 1), 0), out=buf192)
        buf193 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf192, buf193, 49152, grid=grid(49152), stream=stream0)
        buf194 = reinterpret_tensor(buf183, (128, 512), (512, 1), 0); del buf183  # reuse
        # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (128, 384), (384, 1), 0), reinterpret_tensor(arg115_1, (384, 512), (1, 384), 0), out=buf194)
        del arg115_1
        buf196 = reinterpret_tensor(buf180, (1, 128, 512), (65536, 512, 1), 0); del buf180  # reuse
        # Source Nodes: [add_66, hidden_states_122, hidden_states_123, normed_hidden_states_11, pow_31, rsqrt_21, variance_21], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf181, buf194, arg21_1, buf196, 128, 512, grid=grid(128), stream=stream0)
        del arg21_1
        buf197 = reinterpret_tensor(buf193, (128, 384), (384, 1), 0); del buf193  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 384), (1, 512), 0), out=buf197)
        del arg116_1
        buf198 = reinterpret_tensor(buf192, (128, 384), (384, 1), 0); del buf192  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 384), (1, 512), 0), out=buf198)
        del arg117_1
        buf199 = reinterpret_tensor(buf191, (6, 128, 128), (16384, 128, 1), 0); del buf191  # reuse
        # Source Nodes: [scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf197, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf198, (6, 64, 128), (64, 1, 384), 0), out=buf199)
        buf203 = reinterpret_tensor(buf186, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf186  # reuse
        # Source Nodes: [softmax_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf199, buf203, 768, 128, grid=grid(768), stream=stream0)
        buf202 = buf197; del buf197  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg118_1, (512, 384), (1, 512), 0), out=buf202)
        del arg118_1
        buf204 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf203, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf202, (6, 128, 64), (64, 384, 1), 0), out=buf204)
        buf205 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf204, buf205, 49152, grid=grid(49152), stream=stream0)
        buf206 = reinterpret_tensor(buf196, (128, 512), (512, 1), 0); del buf196  # reuse
        # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (128, 384), (384, 1), 0), reinterpret_tensor(arg119_1, (384, 512), (1, 384), 0), out=buf206)
        del arg119_1
        buf208 = reinterpret_tensor(buf174, (1, 128, 512), (65536, 512, 1), 0); del buf174  # reuse
        # Source Nodes: [add_68, forwarded_states_18, hidden_states_122, hidden_states_126, hidden_states_127, pow_32, rsqrt_22, variance_22], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf181, buf194, buf206, arg22_1, buf208, 128, 512, grid=grid(128), stream=stream0)
        del arg22_1
        buf209 = reinterpret_tensor(buf179, (128, 1024), (1024, 1), 0); del buf179  # reuse
        # Source Nodes: [l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (128, 512), (512, 1), 0), reinterpret_tensor(arg120_1, (512, 1024), (1, 512), 0), out=buf209)
        del arg120_1
        buf210 = buf178; del buf178  # reuse
        # Source Nodes: [hidden_linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (128, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 1024), (1, 512), 0), out=buf210)
        del arg121_1
        buf211 = reinterpret_tensor(buf209, (1, 128, 1024), (131072, 1024, 1), 0); del buf209  # reuse
        # Source Nodes: [add_69, add_70, hidden_gelu_9, hidden_states_128, mul_100, mul_98, mul_99, pow_33, tanh_9], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf211, buf210, 131072, grid=grid(131072), stream=stream0)
        buf212 = reinterpret_tensor(buf208, (128, 512), (512, 1), 0); del buf208  # reuse
        # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 512), (1, 1024), 0), out=buf212)
        del arg122_1
        buf214 = reinterpret_tensor(buf138, (1, 128, 512), (65536, 512, 1), 0); del buf138  # reuse
        # Source Nodes: [add_72, hidden_states_122, hidden_states_126, hidden_states_133, hidden_states_134, normed_hidden_states_12, pow_34, rsqrt_23, variance_23], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf181, buf194, buf206, buf212, arg23_1, buf214, 128, 512, grid=grid(128), stream=stream0)
        del arg23_1
        buf215 = reinterpret_tensor(buf205, (128, 384), (384, 1), 0); del buf205  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (128, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 384), (1, 512), 0), out=buf215)
        del arg123_1
        buf216 = reinterpret_tensor(buf204, (128, 384), (384, 1), 0); del buf204  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (128, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 384), (1, 512), 0), out=buf216)
        del arg124_1
        buf217 = reinterpret_tensor(buf203, (6, 128, 128), (16384, 128, 1), 0); del buf203  # reuse
        # Source Nodes: [scores_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf215, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf216, (6, 64, 128), (64, 1, 384), 0), out=buf217)
        buf222 = reinterpret_tensor(buf199, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf199  # reuse
        # Source Nodes: [softmax_12], Original ATen: [aten._softmax]
        triton_per_fused__softmax_1.run(buf217, arg103_1, buf222, 768, 128, grid=grid(768), stream=stream0)
        buf221 = buf215; del buf215  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (128, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 384), (1, 512), 0), out=buf221)
        del arg125_1
        buf223 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf222, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf221, (6, 128, 64), (64, 384, 1), 0), out=buf223)
        buf224 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf223, buf224, 49152, grid=grid(49152), stream=stream0)
        buf225 = reinterpret_tensor(buf214, (128, 512), (512, 1), 0); del buf214  # reuse
        # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (128, 384), (384, 1), 0), reinterpret_tensor(arg126_1, (384, 512), (1, 384), 0), out=buf225)
        del arg126_1
        buf226 = buf181; del buf181  # reuse
        buf228 = reinterpret_tensor(buf126, (1, 128, 512), (65536, 512, 1), 0); del buf126  # reuse
        # Source Nodes: [add_74, hidden_states_122, hidden_states_126, hidden_states_133, hidden_states_138, hidden_states_139, normed_hidden_states_13, pow_35, rsqrt_24, variance_24], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf226, buf194, buf206, buf212, buf225, arg24_1, buf228, 128, 512, grid=grid(128), stream=stream0)
        del arg24_1
        buf229 = reinterpret_tensor(buf224, (128, 384), (384, 1), 0); del buf224  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (128, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 384), (1, 512), 0), out=buf229)
        del arg127_1
        buf230 = reinterpret_tensor(buf223, (128, 384), (384, 1), 0); del buf223  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 384), (1, 512), 0), out=buf230)
        del arg128_1
        buf231 = reinterpret_tensor(buf222, (6, 128, 128), (16384, 128, 1), 0); del buf222  # reuse
        # Source Nodes: [scores_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf230, (6, 64, 128), (64, 1, 384), 0), out=buf231)
        buf235 = reinterpret_tensor(buf217, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf217  # reuse
        # Source Nodes: [softmax_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf231, buf235, 768, 128, grid=grid(768), stream=stream0)
        buf234 = buf229; del buf229  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 384), (1, 512), 0), out=buf234)
        del arg129_1
        buf236 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf234, (6, 128, 64), (64, 384, 1), 0), out=buf236)
        buf237 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf236, buf237, 49152, grid=grid(49152), stream=stream0)
        buf238 = reinterpret_tensor(buf228, (128, 512), (512, 1), 0); del buf228  # reuse
        # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (128, 384), (384, 1), 0), reinterpret_tensor(arg130_1, (384, 512), (1, 384), 0), out=buf238)
        del arg130_1
        buf240 = reinterpret_tensor(buf225, (1, 128, 512), (65536, 512, 1), 0); del buf225  # reuse
        # Source Nodes: [add_76, forwarded_states_20, hidden_states_142, hidden_states_143, pow_36, rsqrt_25, variance_25], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf226, buf238, arg25_1, buf240, 128, 512, grid=grid(128), stream=stream0)
        del arg25_1
        buf241 = reinterpret_tensor(buf211, (128, 1024), (1024, 1), 0); del buf211  # reuse
        # Source Nodes: [l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg131_1, (512, 1024), (1, 512), 0), out=buf241)
        del arg131_1
        buf242 = buf210; del buf210  # reuse
        # Source Nodes: [hidden_linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg132_1, (512, 1024), (1, 512), 0), out=buf242)
        del arg132_1
        buf243 = reinterpret_tensor(buf241, (1, 128, 1024), (131072, 1024, 1), 0); del buf241  # reuse
        # Source Nodes: [add_77, add_78, hidden_gelu_10, hidden_states_144, mul_109, mul_110, mul_111, pow_37, tanh_10], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf243, buf242, 131072, grid=grid(131072), stream=stream0)
        buf244 = reinterpret_tensor(buf240, (128, 512), (512, 1), 0); del buf240  # reuse
        # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 512), (1, 1024), 0), out=buf244)
        del arg133_1
        buf246 = reinterpret_tensor(buf212, (1, 128, 512), (65536, 512, 1), 0); del buf212  # reuse
        # Source Nodes: [add_80, hidden_states_142, hidden_states_149, hidden_states_150, normed_hidden_states_14, pow_38, rsqrt_26, variance_26], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf226, buf238, buf244, arg26_1, buf246, 128, 512, grid=grid(128), stream=stream0)
        del arg26_1
        buf247 = reinterpret_tensor(buf237, (128, 384), (384, 1), 0); del buf237  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (128, 512), (512, 1), 0), reinterpret_tensor(arg134_1, (512, 384), (1, 512), 0), out=buf247)
        del arg134_1
        buf248 = reinterpret_tensor(buf236, (128, 384), (384, 1), 0); del buf236  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (128, 512), (512, 1), 0), reinterpret_tensor(arg135_1, (512, 384), (1, 512), 0), out=buf248)
        del arg135_1
        buf249 = reinterpret_tensor(buf235, (6, 128, 128), (16384, 128, 1), 0); del buf235  # reuse
        # Source Nodes: [scores_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf248, (6, 64, 128), (64, 1, 384), 0), out=buf249)
        buf254 = reinterpret_tensor(buf231, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf231  # reuse
        # Source Nodes: [softmax_14], Original ATen: [aten._softmax]
        triton_per_fused__softmax_1.run(buf249, arg103_1, buf254, 768, 128, grid=grid(768), stream=stream0)
        buf253 = buf247; del buf247  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (128, 512), (512, 1), 0), reinterpret_tensor(arg136_1, (512, 384), (1, 512), 0), out=buf253)
        del arg136_1
        buf255 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf254, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf253, (6, 128, 64), (64, 384, 1), 0), out=buf255)
        buf256 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf255, buf256, 49152, grid=grid(49152), stream=stream0)
        buf257 = reinterpret_tensor(buf246, (128, 512), (512, 1), 0); del buf246  # reuse
        # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (128, 384), (384, 1), 0), reinterpret_tensor(arg137_1, (384, 512), (1, 384), 0), out=buf257)
        del arg137_1
        buf259 = reinterpret_tensor(buf206, (1, 128, 512), (65536, 512, 1), 0); del buf206  # reuse
        # Source Nodes: [add_82, hidden_states_142, hidden_states_149, hidden_states_154, hidden_states_155, normed_hidden_states_15, pow_39, rsqrt_27, variance_27], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf226, buf238, buf244, buf257, arg27_1, buf259, 128, 512, grid=grid(128), stream=stream0)
        del arg27_1
        buf260 = reinterpret_tensor(buf256, (128, 384), (384, 1), 0); del buf256  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (128, 512), (512, 1), 0), reinterpret_tensor(arg138_1, (512, 384), (1, 512), 0), out=buf260)
        del arg138_1
        buf261 = reinterpret_tensor(buf255, (128, 384), (384, 1), 0); del buf255  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg139_1, (512, 384), (1, 512), 0), out=buf261)
        del arg139_1
        buf262 = reinterpret_tensor(buf254, (6, 128, 128), (16384, 128, 1), 0); del buf254  # reuse
        # Source Nodes: [scores_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf260, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf261, (6, 64, 128), (64, 1, 384), 0), out=buf262)
        buf266 = reinterpret_tensor(buf249, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf249  # reuse
        # Source Nodes: [softmax_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf262, buf266, 768, 128, grid=grid(768), stream=stream0)
        buf265 = buf260; del buf260  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg140_1, (512, 384), (1, 512), 0), out=buf265)
        del arg140_1
        buf267 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf266, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf265, (6, 128, 64), (64, 384, 1), 0), out=buf267)
        buf268 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf267, buf268, 49152, grid=grid(49152), stream=stream0)
        buf269 = reinterpret_tensor(buf259, (128, 512), (512, 1), 0); del buf259  # reuse
        # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (128, 384), (384, 1), 0), reinterpret_tensor(arg141_1, (384, 512), (1, 384), 0), out=buf269)
        del arg141_1
        buf270 = buf226; del buf226  # reuse
        buf272 = reinterpret_tensor(buf194, (1, 128, 512), (65536, 512, 1), 0); del buf194  # reuse
        # Source Nodes: [add_84, forwarded_states_22, hidden_states_142, hidden_states_149, hidden_states_154, hidden_states_158, hidden_states_159, pow_40, rsqrt_28, variance_28], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf270, buf238, buf244, buf257, buf269, arg28_1, buf272, 128, 512, grid=grid(128), stream=stream0)
        del arg28_1
        buf273 = reinterpret_tensor(buf243, (128, 1024), (1024, 1), 0); del buf243  # reuse
        # Source Nodes: [l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf272, (128, 512), (512, 1), 0), reinterpret_tensor(arg142_1, (512, 1024), (1, 512), 0), out=buf273)
        del arg142_1
        buf274 = buf242; del buf242  # reuse
        # Source Nodes: [hidden_linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf272, (128, 512), (512, 1), 0), reinterpret_tensor(arg143_1, (512, 1024), (1, 512), 0), out=buf274)
        del arg143_1
        buf275 = reinterpret_tensor(buf273, (1, 128, 1024), (131072, 1024, 1), 0); del buf273  # reuse
        # Source Nodes: [add_85, add_86, hidden_gelu_11, hidden_states_160, mul_120, mul_121, mul_122, pow_41, tanh_11], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf275, buf274, 131072, grid=grid(131072), stream=stream0)
        buf276 = reinterpret_tensor(buf272, (128, 512), (512, 1), 0); del buf272  # reuse
        # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 512), (1, 1024), 0), out=buf276)
        del arg144_1
        buf278 = reinterpret_tensor(buf269, (1, 128, 512), (65536, 512, 1), 0); del buf269  # reuse
        # Source Nodes: [add_88, hidden_states_165, hidden_states_166, normed_hidden_states_16, pow_42, rsqrt_29, variance_29], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf270, buf276, arg29_1, buf278, 128, 512, grid=grid(128), stream=stream0)
        del arg29_1
        buf279 = reinterpret_tensor(buf268, (128, 384), (384, 1), 0); del buf268  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 512), (512, 1), 0), reinterpret_tensor(arg145_1, (512, 384), (1, 512), 0), out=buf279)
        del arg145_1
        buf280 = reinterpret_tensor(buf267, (128, 384), (384, 1), 0); del buf267  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 512), (512, 1), 0), reinterpret_tensor(arg146_1, (512, 384), (1, 512), 0), out=buf280)
        del arg146_1
        buf281 = reinterpret_tensor(buf266, (6, 128, 128), (16384, 128, 1), 0); del buf266  # reuse
        # Source Nodes: [scores_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf280, (6, 64, 128), (64, 1, 384), 0), out=buf281)
        buf286 = reinterpret_tensor(buf262, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf262  # reuse
        # Source Nodes: [softmax_16], Original ATen: [aten._softmax]
        triton_per_fused__softmax_1.run(buf281, arg103_1, buf286, 768, 128, grid=grid(768), stream=stream0)
        buf285 = buf279; del buf279  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 512), (512, 1), 0), reinterpret_tensor(arg147_1, (512, 384), (1, 512), 0), out=buf285)
        del arg147_1
        buf287 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf286, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf285, (6, 128, 64), (64, 384, 1), 0), out=buf287)
        buf288 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf287, buf288, 49152, grid=grid(49152), stream=stream0)
        buf289 = reinterpret_tensor(buf278, (128, 512), (512, 1), 0); del buf278  # reuse
        # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (128, 384), (384, 1), 0), reinterpret_tensor(arg148_1, (384, 512), (1, 384), 0), out=buf289)
        del arg148_1
        buf291 = reinterpret_tensor(buf257, (1, 128, 512), (65536, 512, 1), 0); del buf257  # reuse
        # Source Nodes: [add_90, hidden_states_165, hidden_states_170, hidden_states_171, normed_hidden_states_17, pow_43, rsqrt_30, variance_30], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf270, buf276, buf289, arg30_1, buf291, 128, 512, grid=grid(128), stream=stream0)
        del arg30_1
        buf292 = reinterpret_tensor(buf288, (128, 384), (384, 1), 0); del buf288  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (128, 512), (512, 1), 0), reinterpret_tensor(arg149_1, (512, 384), (1, 512), 0), out=buf292)
        del arg149_1
        buf293 = reinterpret_tensor(buf287, (128, 384), (384, 1), 0); del buf287  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg150_1, (512, 384), (1, 512), 0), out=buf293)
        del arg150_1
        buf294 = reinterpret_tensor(buf286, (6, 128, 128), (16384, 128, 1), 0); del buf286  # reuse
        # Source Nodes: [scores_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf293, (6, 64, 128), (64, 1, 384), 0), out=buf294)
        buf298 = reinterpret_tensor(buf281, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf281  # reuse
        # Source Nodes: [softmax_17], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf294, buf298, 768, 128, grid=grid(768), stream=stream0)
        buf297 = buf292; del buf292  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg151_1, (512, 384), (1, 512), 0), out=buf297)
        del arg151_1
        buf299 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf298, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf297, (6, 128, 64), (64, 384, 1), 0), out=buf299)
        buf300 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf299, buf300, 49152, grid=grid(49152), stream=stream0)
        buf301 = reinterpret_tensor(buf291, (128, 512), (512, 1), 0); del buf291  # reuse
        # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (128, 384), (384, 1), 0), reinterpret_tensor(arg152_1, (384, 512), (1, 384), 0), out=buf301)
        del arg152_1
        buf303 = reinterpret_tensor(buf244, (1, 128, 512), (65536, 512, 1), 0); del buf244  # reuse
        # Source Nodes: [add_92, forwarded_states_24, hidden_states_165, hidden_states_170, hidden_states_174, hidden_states_175, pow_44, rsqrt_31, variance_31], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf270, buf276, buf289, buf301, arg31_1, buf303, 128, 512, grid=grid(128), stream=stream0)
        del arg31_1
        buf304 = reinterpret_tensor(buf275, (128, 1024), (1024, 1), 0); del buf275  # reuse
        # Source Nodes: [l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (128, 512), (512, 1), 0), reinterpret_tensor(arg153_1, (512, 1024), (1, 512), 0), out=buf304)
        del arg153_1
        buf305 = buf274; del buf274  # reuse
        # Source Nodes: [hidden_linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (128, 512), (512, 1), 0), reinterpret_tensor(arg154_1, (512, 1024), (1, 512), 0), out=buf305)
        del arg154_1
        buf306 = reinterpret_tensor(buf304, (1, 128, 1024), (131072, 1024, 1), 0); del buf304  # reuse
        # Source Nodes: [add_93, add_94, hidden_gelu_12, hidden_states_176, mul_131, mul_132, mul_133, pow_45, tanh_12], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf306, buf305, 131072, grid=grid(131072), stream=stream0)
        buf307 = reinterpret_tensor(buf303, (128, 512), (512, 1), 0); del buf303  # reuse
        # Source Nodes: [forwarded_states_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf306, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 512), (1, 1024), 0), out=buf307)
        del arg155_1
        buf308 = buf270; del buf270  # reuse
        buf310 = reinterpret_tensor(buf238, (1, 128, 512), (65536, 512, 1), 0); del buf238  # reuse
        # Source Nodes: [add_96, hidden_states_165, hidden_states_170, hidden_states_174, hidden_states_181, hidden_states_182, normed_hidden_states_18, pow_46, rsqrt_32, variance_32], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf308, buf276, buf289, buf301, buf307, arg32_1, buf310, 128, 512, grid=grid(128), stream=stream0)
        del arg32_1
        buf311 = reinterpret_tensor(buf300, (128, 384), (384, 1), 0); del buf300  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 512), (512, 1), 0), reinterpret_tensor(arg156_1, (512, 384), (1, 512), 0), out=buf311)
        del arg156_1
        buf312 = reinterpret_tensor(buf299, (128, 384), (384, 1), 0); del buf299  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 512), (512, 1), 0), reinterpret_tensor(arg157_1, (512, 384), (1, 512), 0), out=buf312)
        del arg157_1
        buf313 = reinterpret_tensor(buf298, (6, 128, 128), (16384, 128, 1), 0); del buf298  # reuse
        # Source Nodes: [scores_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf311, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf312, (6, 64, 128), (64, 1, 384), 0), out=buf313)
        buf318 = reinterpret_tensor(buf294, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf294  # reuse
        # Source Nodes: [softmax_18], Original ATen: [aten._softmax]
        triton_per_fused__softmax_1.run(buf313, arg103_1, buf318, 768, 128, grid=grid(768), stream=stream0)
        buf317 = buf311; del buf311  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 512), (512, 1), 0), reinterpret_tensor(arg158_1, (512, 384), (1, 512), 0), out=buf317)
        del arg158_1
        buf319 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf318, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf317, (6, 128, 64), (64, 384, 1), 0), out=buf319)
        buf320 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf319, buf320, 49152, grid=grid(49152), stream=stream0)
        buf321 = reinterpret_tensor(buf310, (128, 512), (512, 1), 0); del buf310  # reuse
        # Source Nodes: [attn_output_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (128, 384), (384, 1), 0), reinterpret_tensor(arg159_1, (384, 512), (1, 384), 0), out=buf321)
        del arg159_1
        buf323 = reinterpret_tensor(buf307, (1, 128, 512), (65536, 512, 1), 0); del buf307  # reuse
        # Source Nodes: [add_98, hidden_states_186, hidden_states_187, normed_hidden_states_19, pow_47, rsqrt_33, variance_33], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf308, buf321, arg33_1, buf323, 128, 512, grid=grid(128), stream=stream0)
        del arg33_1
        buf324 = reinterpret_tensor(buf320, (128, 384), (384, 1), 0); del buf320  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (128, 512), (512, 1), 0), reinterpret_tensor(arg160_1, (512, 384), (1, 512), 0), out=buf324)
        del arg160_1
        buf325 = reinterpret_tensor(buf319, (128, 384), (384, 1), 0); del buf319  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg161_1, (512, 384), (1, 512), 0), out=buf325)
        del arg161_1
        buf326 = reinterpret_tensor(buf318, (6, 128, 128), (16384, 128, 1), 0); del buf318  # reuse
        # Source Nodes: [scores_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf325, (6, 64, 128), (64, 1, 384), 0), out=buf326)
        buf330 = reinterpret_tensor(buf313, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf313  # reuse
        # Source Nodes: [softmax_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf326, buf330, 768, 128, grid=grid(768), stream=stream0)
        buf329 = buf324; del buf324  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg162_1, (512, 384), (1, 512), 0), out=buf329)
        del arg162_1
        buf331 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf329, (6, 128, 64), (64, 384, 1), 0), out=buf331)
        buf332 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf331, buf332, 49152, grid=grid(49152), stream=stream0)
        buf333 = reinterpret_tensor(buf323, (128, 512), (512, 1), 0); del buf323  # reuse
        # Source Nodes: [attn_output_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 384), (384, 1), 0), reinterpret_tensor(arg163_1, (384, 512), (1, 384), 0), out=buf333)
        del arg163_1
        buf335 = reinterpret_tensor(buf301, (1, 128, 512), (65536, 512, 1), 0); del buf301  # reuse
        # Source Nodes: [add_100, forwarded_states_26, hidden_states_186, hidden_states_190, hidden_states_191, pow_48, rsqrt_34, variance_34], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf308, buf321, buf333, arg34_1, buf335, 128, 512, grid=grid(128), stream=stream0)
        del arg34_1
        buf336 = reinterpret_tensor(buf306, (128, 1024), (1024, 1), 0); del buf306  # reuse
        # Source Nodes: [l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (128, 512), (512, 1), 0), reinterpret_tensor(arg164_1, (512, 1024), (1, 512), 0), out=buf336)
        del arg164_1
        buf337 = buf305; del buf305  # reuse
        # Source Nodes: [hidden_linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (128, 512), (512, 1), 0), reinterpret_tensor(arg165_1, (512, 1024), (1, 512), 0), out=buf337)
        del arg165_1
        buf338 = reinterpret_tensor(buf336, (1, 128, 1024), (131072, 1024, 1), 0); del buf336  # reuse
        # Source Nodes: [add_101, add_102, hidden_gelu_13, hidden_states_192, mul_142, mul_143, mul_144, pow_49, tanh_13], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf338, buf337, 131072, grid=grid(131072), stream=stream0)
        buf339 = reinterpret_tensor(buf335, (128, 512), (512, 1), 0); del buf335  # reuse
        # Source Nodes: [forwarded_states_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 512), (1, 1024), 0), out=buf339)
        del arg166_1
        buf341 = reinterpret_tensor(buf289, (1, 128, 512), (65536, 512, 1), 0); del buf289  # reuse
        # Source Nodes: [add_104, hidden_states_186, hidden_states_190, hidden_states_197, hidden_states_198, normed_hidden_states_20, pow_50, rsqrt_35, variance_35], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf308, buf321, buf333, buf339, arg35_1, buf341, 128, 512, grid=grid(128), stream=stream0)
        del arg35_1
        buf342 = reinterpret_tensor(buf332, (128, 384), (384, 1), 0); del buf332  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (128, 512), (512, 1), 0), reinterpret_tensor(arg167_1, (512, 384), (1, 512), 0), out=buf342)
        del arg167_1
        buf343 = reinterpret_tensor(buf331, (128, 384), (384, 1), 0); del buf331  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (128, 512), (512, 1), 0), reinterpret_tensor(arg168_1, (512, 384), (1, 512), 0), out=buf343)
        del arg168_1
        buf344 = reinterpret_tensor(buf330, (6, 128, 128), (16384, 128, 1), 0); del buf330  # reuse
        # Source Nodes: [scores_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf342, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf343, (6, 64, 128), (64, 1, 384), 0), out=buf344)
        buf349 = reinterpret_tensor(buf326, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf326  # reuse
        # Source Nodes: [softmax_20], Original ATen: [aten._softmax]
        triton_per_fused__softmax_1.run(buf344, arg103_1, buf349, 768, 128, grid=grid(768), stream=stream0)
        buf348 = buf342; del buf342  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (128, 512), (512, 1), 0), reinterpret_tensor(arg169_1, (512, 384), (1, 512), 0), out=buf348)
        del arg169_1
        buf350 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf349, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf348, (6, 128, 64), (64, 384, 1), 0), out=buf350)
        buf351 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf350, buf351, 49152, grid=grid(49152), stream=stream0)
        buf352 = reinterpret_tensor(buf341, (128, 512), (512, 1), 0); del buf341  # reuse
        # Source Nodes: [attn_output_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (128, 384), (384, 1), 0), reinterpret_tensor(arg170_1, (384, 512), (1, 384), 0), out=buf352)
        del arg170_1
        buf353 = buf308; del buf308  # reuse
        buf355 = reinterpret_tensor(buf276, (1, 128, 512), (65536, 512, 1), 0); del buf276  # reuse
        # Source Nodes: [add_106, hidden_states_186, hidden_states_190, hidden_states_197, hidden_states_202, hidden_states_203, normed_hidden_states_21, pow_51, rsqrt_36, variance_36], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf353, buf321, buf333, buf339, buf352, arg36_1, buf355, 128, 512, grid=grid(128), stream=stream0)
        del arg36_1
        buf356 = reinterpret_tensor(buf351, (128, 384), (384, 1), 0); del buf351  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (128, 512), (512, 1), 0), reinterpret_tensor(arg171_1, (512, 384), (1, 512), 0), out=buf356)
        del arg171_1
        buf357 = reinterpret_tensor(buf350, (128, 384), (384, 1), 0); del buf350  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg172_1, (512, 384), (1, 512), 0), out=buf357)
        del arg172_1
        buf358 = reinterpret_tensor(buf349, (6, 128, 128), (16384, 128, 1), 0); del buf349  # reuse
        # Source Nodes: [scores_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf356, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf357, (6, 64, 128), (64, 1, 384), 0), out=buf358)
        buf362 = reinterpret_tensor(buf344, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf344  # reuse
        # Source Nodes: [softmax_21], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf358, buf362, 768, 128, grid=grid(768), stream=stream0)
        buf361 = buf356; del buf356  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg173_1, (512, 384), (1, 512), 0), out=buf361)
        del arg173_1
        buf363 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf361, (6, 128, 64), (64, 384, 1), 0), out=buf363)
        buf364 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf363, buf364, 49152, grid=grid(49152), stream=stream0)
        buf365 = reinterpret_tensor(buf355, (128, 512), (512, 1), 0); del buf355  # reuse
        # Source Nodes: [attn_output_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (128, 384), (384, 1), 0), reinterpret_tensor(arg174_1, (384, 512), (1, 384), 0), out=buf365)
        del arg174_1
        buf367 = reinterpret_tensor(buf352, (1, 128, 512), (65536, 512, 1), 0); del buf352  # reuse
        # Source Nodes: [add_108, forwarded_states_28, hidden_states_206, hidden_states_207, pow_52, rsqrt_37, variance_37], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf353, buf365, arg37_1, buf367, 128, 512, grid=grid(128), stream=stream0)
        del arg37_1
        buf368 = reinterpret_tensor(buf338, (128, 1024), (1024, 1), 0); del buf338  # reuse
        # Source Nodes: [l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (128, 512), (512, 1), 0), reinterpret_tensor(arg175_1, (512, 1024), (1, 512), 0), out=buf368)
        del arg175_1
        buf369 = buf337; del buf337  # reuse
        # Source Nodes: [hidden_linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (128, 512), (512, 1), 0), reinterpret_tensor(arg176_1, (512, 1024), (1, 512), 0), out=buf369)
        del arg176_1
        buf370 = reinterpret_tensor(buf368, (1, 128, 1024), (131072, 1024, 1), 0); del buf368  # reuse
        # Source Nodes: [add_109, add_110, hidden_gelu_14, hidden_states_208, mul_153, mul_154, mul_155, pow_53, tanh_14], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf370, buf369, 131072, grid=grid(131072), stream=stream0)
        buf371 = reinterpret_tensor(buf367, (128, 512), (512, 1), 0); del buf367  # reuse
        # Source Nodes: [forwarded_states_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg177_1, (1024, 512), (1, 1024), 0), out=buf371)
        del arg177_1
        buf373 = reinterpret_tensor(buf339, (1, 128, 512), (65536, 512, 1), 0); del buf339  # reuse
        # Source Nodes: [add_112, hidden_states_206, hidden_states_213, hidden_states_214, normed_hidden_states_22, pow_54, rsqrt_38, variance_38], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_10.run(buf353, buf365, buf371, arg38_1, buf373, 128, 512, grid=grid(128), stream=stream0)
        del arg38_1
        buf374 = reinterpret_tensor(buf364, (128, 384), (384, 1), 0); del buf364  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (128, 512), (512, 1), 0), reinterpret_tensor(arg178_1, (512, 384), (1, 512), 0), out=buf374)
        del arg178_1
        buf375 = reinterpret_tensor(buf363, (128, 384), (384, 1), 0); del buf363  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (128, 512), (512, 1), 0), reinterpret_tensor(arg179_1, (512, 384), (1, 512), 0), out=buf375)
        del arg179_1
        buf376 = reinterpret_tensor(buf362, (6, 128, 128), (16384, 128, 1), 0); del buf362  # reuse
        # Source Nodes: [scores_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf374, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf375, (6, 64, 128), (64, 1, 384), 0), out=buf376)
        buf381 = reinterpret_tensor(buf358, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf358  # reuse
        # Source Nodes: [softmax_22], Original ATen: [aten._softmax]
        triton_per_fused__softmax_1.run(buf376, arg103_1, buf381, 768, 128, grid=grid(768), stream=stream0)
        del arg103_1
        buf380 = buf374; del buf374  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (128, 512), (512, 1), 0), reinterpret_tensor(arg180_1, (512, 384), (1, 512), 0), out=buf380)
        del arg180_1
        buf382 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf381, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf380, (6, 128, 64), (64, 384, 1), 0), out=buf382)
        buf383 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf382, buf383, 49152, grid=grid(49152), stream=stream0)
        buf384 = reinterpret_tensor(buf373, (128, 512), (512, 1), 0); del buf373  # reuse
        # Source Nodes: [attn_output_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (128, 384), (384, 1), 0), reinterpret_tensor(arg181_1, (384, 512), (1, 384), 0), out=buf384)
        del arg181_1
        buf386 = reinterpret_tensor(buf333, (1, 128, 512), (65536, 512, 1), 0); del buf333  # reuse
        # Source Nodes: [add_114, hidden_states_206, hidden_states_213, hidden_states_218, hidden_states_219, normed_hidden_states_23, pow_55, rsqrt_39, variance_39], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf353, buf365, buf371, buf384, arg39_1, buf386, 128, 512, grid=grid(128), stream=stream0)
        del arg39_1
        buf387 = reinterpret_tensor(buf383, (128, 384), (384, 1), 0); del buf383  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (128, 512), (512, 1), 0), reinterpret_tensor(arg182_1, (512, 384), (1, 512), 0), out=buf387)
        del arg182_1
        buf388 = reinterpret_tensor(buf382, (128, 384), (384, 1), 0); del buf382  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg183_1, (512, 384), (1, 512), 0), out=buf388)
        del arg183_1
        buf389 = reinterpret_tensor(buf381, (6, 128, 128), (16384, 128, 1), 0); del buf381  # reuse
        # Source Nodes: [scores_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf387, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf388, (6, 64, 128), (64, 1, 384), 0), out=buf389)
        buf393 = reinterpret_tensor(buf376, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf376  # reuse
        # Source Nodes: [softmax_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_14.run(buf389, buf393, 768, 128, grid=grid(768), stream=stream0)
        del buf389
        buf392 = buf387; del buf387  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg184_1, (512, 384), (1, 512), 0), out=buf392)
        del arg184_1
        buf394 = empty((6, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf393, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf392, (6, 128, 64), (64, 384, 1), 0), out=buf394)
        del buf393
        buf395 = empty((1, 128, 6, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf394, buf395, 49152, grid=grid(49152), stream=stream0)
        del buf394
        buf396 = reinterpret_tensor(buf386, (128, 512), (512, 1), 0); del buf386  # reuse
        # Source Nodes: [attn_output_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (128, 384), (384, 1), 0), reinterpret_tensor(arg185_1, (384, 512), (1, 384), 0), out=buf396)
        del arg185_1
        del buf395
        buf397 = buf353; del buf353  # reuse
        buf399 = reinterpret_tensor(buf321, (1, 128, 512), (65536, 512, 1), 0); del buf321  # reuse
        # Source Nodes: [add_116, forwarded_states_30, hidden_states_206, hidden_states_213, hidden_states_218, hidden_states_222, hidden_states_223, pow_56, rsqrt_40, variance_40], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf397, buf365, buf371, buf384, buf396, arg40_1, buf399, 128, 512, grid=grid(128), stream=stream0)
        del arg40_1
        del buf365
        del buf371
        del buf384
        buf400 = reinterpret_tensor(buf370, (128, 1024), (1024, 1), 0); del buf370  # reuse
        # Source Nodes: [l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (128, 512), (512, 1), 0), reinterpret_tensor(arg186_1, (512, 1024), (1, 512), 0), out=buf400)
        del arg186_1
        buf401 = buf369; del buf369  # reuse
        # Source Nodes: [hidden_linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (128, 512), (512, 1), 0), reinterpret_tensor(arg187_1, (512, 1024), (1, 512), 0), out=buf401)
        del arg187_1
        buf402 = reinterpret_tensor(buf400, (1, 128, 1024), (131072, 1024, 1), 0); del buf400  # reuse
        # Source Nodes: [add_117, add_118, hidden_gelu_15, hidden_states_224, mul_164, mul_165, mul_166, pow_57, tanh_15], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
        triton_poi_fused_add_mul_pow_tanh_6.run(buf402, buf401, 131072, grid=grid(131072), stream=stream0)
        del buf401
        buf403 = reinterpret_tensor(buf399, (128, 512), (512, 1), 0); del buf399  # reuse
        # Source Nodes: [forwarded_states_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf402, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 512), (1, 1024), 0), out=buf403)
        del arg188_1
        del buf402
        buf405 = reinterpret_tensor(buf396, (1, 128, 512), (65536, 512, 1), 0); del buf396  # reuse
        # Source Nodes: [add_120, hidden_states_229, hidden_states_230, hidden_states_231, pow_58, rsqrt_41, variance_41], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf397, buf403, arg41_1, buf405, 128, 512, grid=grid(128), stream=stream0)
        del arg41_1
        del buf397
        del buf403
        buf406 = empty((128, 250112), device='cuda', dtype=torch.float32)
        # Source Nodes: [lm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (128, 512), (512, 1), 0), reinterpret_tensor(arg189_1, (512, 250112), (1, 512), 0), out=buf406)
        del arg189_1
        del buf405
        buf407 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_15.run(buf406, buf407, 512, 62528, grid=grid(512), stream=stream0)
        buf408 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_16.run(buf407, buf408, 128, 4, grid=grid(128), stream=stream0)
        buf409 = buf407; del buf407  # reuse
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_17.run(buf406, buf408, buf409, 512, 62528, grid=grid(512), stream=stream0)
        buf410 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_per_fused__log_softmax_18.run(buf409, buf410, 128, 4, grid=grid(128), stream=stream0)
        del buf409
        buf411 = empty((), device='cuda', dtype=torch.float32)
        buf413 = buf411; del buf411  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_19.run(buf413, arg191_1, buf406, buf408, buf410, 1, 128, grid=grid(1), stream=stream0)
        del arg191_1
        return (buf413, reinterpret_tensor(buf406, (1, 128, 250112), (32014336, 250112, 1), 0), reinterpret_tensor(buf3, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf8, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf166, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf170, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf185, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf190, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf198, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf202, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf216, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf221, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf230, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf234, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf248, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf253, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf261, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf265, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf280, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf285, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf293, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf297, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf312, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf317, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf325, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf329, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf343, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf348, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf357, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf361, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf375, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf380, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf388, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf392, (1, 6, 128, 64), (49152, 64, 384, 1), 0), buf165, )


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
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((250112, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((32, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((32, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((250112, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg191_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg192_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MT5ForConditionalGeneration', benchmark_compiled_module)
