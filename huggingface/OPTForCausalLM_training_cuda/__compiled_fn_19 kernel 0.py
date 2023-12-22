
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


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tfputiwnft7mzr26aqnmom74n55ioct2lng3kcmspe5cr7vnbt.py
# Source Nodes: [hidden_states, l__self___self_attn_q_proj], Original ATen: [aten.native_layer_norm, aten.view]
# hidden_states => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# l__self___self_attn_q_proj => view
triton_per_fused_native_layer_norm_view_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_0', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 768.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, None)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp27, rmask)
    tl.store(out_ptr0 + (x0), tmp10, None)
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


# kernel path: /tmp/torchinductor_youkaichao/ch/cchddldqc62dykosm4vssoofvlfenip6xe3t375x55bapm377t3c.py
# Source Nodes: [attn_probs, attn_weights_1, attn_weights_4, tensor], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach, aten.eq, aten.lift_fresh, aten.lt]
# attn_probs => clone_3
# attn_weights_1 => add_2
# attn_weights_4 => amax, div, exp, sub_1, sum_1
# tensor => full_default
triton_red_fused__softmax_add_clone_detach_eq_lift_fresh_lt_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_clone_detach_eq_lift_fresh_lt_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    _tmp6 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = -3.4028234663852886e+38
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = triton_helpers.maximum(_tmp6, tmp5)
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = triton_helpers.max2(_tmp6, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr1 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 + tmp9
        tmp11 = -3.4028234663852886e+38
        tmp12 = triton_helpers.maximum(tmp10, tmp11)
        tmp13 = tmp12 - tmp6
        tmp14 = tl.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr1 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = -3.4028234663852886e+38
        tmp22 = triton_helpers.maximum(tmp20, tmp21)
        tmp23 = tmp22 - tmp6
        tmp24 = tl.exp(tmp23)
        tmp25 = tmp24 / tmp16
        tmp26 = tmp20 == tmp21
        tmp27 = tmp20 < tmp21
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp25, rmask)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp25, rmask)
        tl.store(out_ptr4 + (r2 + (2048*x3)), tmp26, rmask)
        tl.store(out_ptr5 + (r2 + (2048*x3)), tmp27, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5g2qf7msmy6zypr4xpy5vvg744sa7ygamjj4olfrdfpmkmuxgxf.py
# Source Nodes: [hidden_states_1], Original ATen: [aten.view]
# hidden_states_1 => view_16
triton_poi_fused_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (131072*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zp/czp6jot4tgec7e5cnawougx7snzptqsll62p2r2mpobog4kyq22b.py
# Source Nodes: [hidden_states_5], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# hidden_states_5 => add_4, add_5, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
triton_per_fused_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp29, rmask)
    tl.store(out_ptr4 + (x0), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55uapboz3i6pmsr6ixn2koe7tybbunkmaazj66ner22nj4e6qew.py
# Source Nodes: [hidden_states_7], Original ATen: [aten.relu]
# hidden_states_7 => relu
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


# kernel path: /tmp/torchinductor_youkaichao/rg/crgznn3asywakdob3ddgo6qp2adw77y5zboqwmihkds2dgqcwu7t.py
# Source Nodes: [add_2, hidden_states_10], Original ATen: [aten.add, aten.view]
# add_2 => add_6
# hidden_states_10 => view_19
triton_poi_fused_add_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_view_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, 768), (768, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 768), (768, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, 768), (768, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, 768), (768, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (3072, 768), (768, 1))
    assert_size_stride(primals_14, (3072, ), (1, ))
    assert_size_stride(primals_15, (768, 3072), (3072, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (1, 2048, 768), (1572864, 768, 1))
    assert_size_stride(primals_18, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 2048, 1), device='cuda', dtype=torch.float32)
        buf1 = empty_strided((1, 2048, 1), (2048, 1, 2048), device='cuda', dtype=torch.float32)
        buf3 = reinterpret_tensor(buf1, (1, 2048, 1), (2048, 1, 1), 0); del buf1  # reuse
        buf4 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states, l__self___self_attn_q_proj], Original ATen: [aten.native_layer_norm, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_layer_norm_view_0.run(buf3, primals_17, primals_1, primals_2, buf0, buf4, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_2
        buf5 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(primals_3, (768, 768), (1, 768), 0), out=buf5)
        buf6 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(primals_5, (768, 768), (1, 768), 0), out=buf6)
        buf7 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf6, primals_6, buf7, 1572864, grid=grid(1572864), stream=stream0)
        del primals_6
        buf8 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(primals_7, (768, 768), (1, 768), 0), out=buf8)
        buf9 = empty((1, 12, 2048, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf8, primals_8, buf9, 1572864, grid=grid(1572864), stream=stream0)
        del primals_8
        buf10 = reinterpret_tensor(buf8, (1, 12, 2048, 64), (1572864, 131072, 64, 1), 0); del buf8  # reuse
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf5, primals_4, buf10, 1572864, grid=grid(1572864), stream=stream0)
        del primals_4
        buf11 = empty((12, 2048, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf7, (12, 64, 2048), (131072, 1, 64), 0), out=buf11)
        buf14 = empty((12, 2048, 2048), device='cuda', dtype=torch.float32)
        buf34 = empty((12, 2048, 2048), device='cuda', dtype=torch.float32)
        buf35 = empty((1, 12, 2048, 2048), device='cuda', dtype=torch.bool)
        buf36 = empty((1, 12, 2048, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [attn_probs, attn_weights_1, attn_weights_4, tensor], Original ATen: [aten._softmax, aten.add, aten.clone, aten.detach, aten.eq, aten.lift_fresh, aten.lt]
        triton_red_fused__softmax_add_clone_detach_eq_lift_fresh_lt_3.run(buf11, primals_18, buf14, buf34, buf35, buf36, 24576, 2048, grid=grid(24576), stream=stream0)
        del buf11
        del primals_18
        buf15 = reinterpret_tensor(buf5, (12, 2048, 64), (131072, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf14, reinterpret_tensor(buf9, (12, 2048, 64), (131072, 64, 1), 0), out=buf15)
        buf16 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf15, buf16, 1572864, grid=grid(1572864), stream=stream0)
        buf17 = reinterpret_tensor(buf15, (2048, 768), (768, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, buf16, reinterpret_tensor(primals_9, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf17)
        del primals_10
        # Source Nodes: [hidden_states_2], Original ATen: [aten.native_dropout]
        buf18 = aten.native_dropout(reinterpret_tensor(buf17, (1, 2048, 768), (1572864, 768, 1), 0), 0.1, True)
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf24 = buf17; del buf17  # reuse
        buf25 = empty((2048, 768), device='cuda', dtype=torch.float32)
        buf33 = empty((2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_5], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_5.run(primals_17, buf19, primals_11, primals_12, buf24, buf25, buf33, 2048, 768, grid=grid(2048), stream=stream0)
        del primals_12
        buf26 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf25, reinterpret_tensor(primals_13, (768, 3072), (1, 768), 0), out=buf26)
        buf27 = buf26; del buf26  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf27, primals_14, 6291456, grid=grid(6291456), stream=stream0)
        del primals_14
        buf28 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_16, buf27, reinterpret_tensor(primals_15, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf28)
        del primals_16
        # Source Nodes: [hidden_states_9], Original ATen: [aten.native_dropout]
        buf29 = aten.native_dropout(buf28, 0.1, True)
        del buf28
        buf30 = buf29[0]
        buf31 = buf29[1]
        del buf29
        buf32 = reinterpret_tensor(buf30, (1, 2048, 768), (1572864, 768, 1), 0); del buf30  # reuse
        # Source Nodes: [add_2, hidden_states_10], Original ATen: [aten.add, aten.view]
        triton_poi_fused_add_view_7.run(buf32, primals_17, buf19, 1572864, grid=grid(1572864), stream=stream0)
        return (buf32, buf7, buf9, primals_1, primals_11, primals_17, buf0, buf3, buf4, buf16, buf20, buf24, buf25, buf27, buf31, reinterpret_tensor(primals_15, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_13, (3072, 768), (768, 1), 0), buf33, reinterpret_tensor(primals_9, (768, 768), (768, 1), 0), reinterpret_tensor(buf14, (12, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf9, (12, 64, 2048), (131072, 1, 64), 0), buf34, buf35, buf36, reinterpret_tensor(buf10, (12, 64, 2048), (131072, 1, 64), 0), reinterpret_tensor(buf7, (12, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(primals_7, (768, 768), (768, 1), 0), reinterpret_tensor(primals_5, (768, 768), (768, 1), 0), reinterpret_tensor(primals_3, (768, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
