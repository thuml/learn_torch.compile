
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


# kernel path: /tmp/torchinductor_youkaichao/kr/ckr4pxuy42ao3o6tjfnqbcve2rtkmmx35mgtbrp66372fvt2j23g.py
# Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# contiguous_2 => clone_2
triton_poi_fused_clone_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/xz/cxzxelroobrwpdtjldpce7vzu2l7huvhtzoja7f2lkb5lb5vi5kh.py
# Source Nodes: [value_states], Original ATen: [aten.clone]
# value_states => clone_1
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwqewipaqbt45xhwzshc5egwyv3fqo3won7x5cvobpcrgswmzn5q.py
# Source Nodes: [attn_probs, attn_weights_3], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_probs => clone_3
# attn_weights_3 => amax, div, exp, sub, sum_1
triton_per_fused__softmax_clone_detach_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp5, 0))
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp13, rmask)
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u7/cu7xhpcddwudgv3ralk7l4tgwzp5yr3rerrz7rk2ktnwkt4d75bq.py
# Source Nodes: [hidden_states], Original ATen: [aten.view]
# hidden_states => view_16
triton_poi_fused_view_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (65536*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxqt25xqfkhslflv6uxjennh36zdo5azx6cawuas4sq43ygwnel.py
# Source Nodes: [hidden_states_2, l__self___encoder_attn_q_proj, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_2 => add_1
# l__self___encoder_attn_q_proj => view_18
# residual_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 1024.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chy6phiaehsex33bgnl2544ziuw4t3ikqli4llh2jf2ayy3xe34b.py
# Source Nodes: [attn_weights_5], Original ATen: [aten._softmax]
# attn_weights_5 => amax_1, div_1, exp_1, sub_2, sum_2
triton_per_fused__softmax_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 16384
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
    tl.store(out_ptr0 + (x0), tmp4, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zjom5rcypybayicqs42wyx5c6jg7wlfn4rf6h5czjaydvkwbdt.py
# Source Nodes: [hidden_states_6, l__self___fc1, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_6 => add_4
# l__self___fc1 => view_34
# residual_1 => add_3, mul_2
# residual_2 => add_5, add_6, mul_4, mul_5, rsqrt_1, sub_3, var_mean_1
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 1024, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 1024.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 / tmp24
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7l72cnccifphpcuu2o2genhcq6vhcijxlj2oc7zwhc4rdrniavs.py
# Source Nodes: [hidden_states_10, hidden_states_8], Original ATen: [aten.gelu, aten.view]
# hidden_states_10 => view_36
# hidden_states_8 => add_7, erf, mul_6, mul_7, mul_8
triton_poi_fused_gelu_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29 = args
    args.clear()
    assert_size_stride(primals_1, (1024, 1024), (1024, 1))
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_3, (1024, 1024), (1024, 1))
    assert_size_stride(primals_4, (1024, ), (1, ))
    assert_size_stride(primals_5, (1024, 1024), (1024, 1))
    assert_size_stride(primals_6, (1024, ), (1, ))
    assert_size_stride(primals_7, (1024, 1024), (1024, 1))
    assert_size_stride(primals_8, (1024, ), (1, ))
    assert_size_stride(primals_9, (1024, ), (1, ))
    assert_size_stride(primals_10, (1024, ), (1, ))
    assert_size_stride(primals_11, (1024, 1024), (1024, 1))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_13, (1024, 1024), (1024, 1))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_15, (1024, 1024), (1024, 1))
    assert_size_stride(primals_16, (1024, ), (1, ))
    assert_size_stride(primals_17, (1024, 1024), (1024, 1))
    assert_size_stride(primals_18, (1024, ), (1, ))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_20, (1024, ), (1, ))
    assert_size_stride(primals_21, (4096, 1024), (1024, 1))
    assert_size_stride(primals_22, (4096, ), (1, ))
    assert_size_stride(primals_23, (1024, 4096), (4096, 1))
    assert_size_stride(primals_24, (1024, ), (1, ))
    assert_size_stride(primals_25, (1024, ), (1, ))
    assert_size_stride(primals_26, (1024, ), (1, ))
    assert_size_stride(primals_27, (1, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(primals_28, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_29, (1, 1024, 1024), (1048576, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_1, (1024, 1024), (1, 1024), 0), out=buf0)
        buf1 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_3, (1024, 1024), (1, 1024), 0), out=buf1)
        buf2 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_27, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_5, (1024, 1024), (1, 1024), 0), out=buf2)
        buf3 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_clone_0.run(buf0, primals_2, buf3, 1048576, grid=grid(1048576), stream=stream0)
        del primals_2
        buf4 = reinterpret_tensor(buf0, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf0  # reuse
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf2, primals_6, buf4, 1048576, grid=grid(1048576), stream=stream0)
        del primals_6
        buf5 = reinterpret_tensor(buf2, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf2  # reuse
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf1, primals_4, buf5, 1048576, grid=grid(1048576), stream=stream0)
        del primals_4
        buf6 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf5, (16, 64, 1024), (65536, 1, 64), 0), out=buf6)
        buf9 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        buf56 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_probs, attn_weights_3], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_2.run(buf6, primals_28, buf9, buf56, 16384, 1024, grid=grid(16384), stream=stream0)
        del primals_28
        buf10 = reinterpret_tensor(buf1, (16, 1024, 64), (65536, 64, 1), 0); del buf1  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf9, reinterpret_tensor(buf4, (16, 1024, 64), (65536, 64, 1), 0), out=buf10)
        buf11 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf10, buf11, 1048576, grid=grid(1048576), stream=stream0)
        buf12 = reinterpret_tensor(buf10, (1024, 1024), (1024, 1), 0); del buf10  # reuse
        # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf11, reinterpret_tensor(primals_7, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf12)
        del primals_8
        # Source Nodes: [hidden_states_1], Original ATen: [aten.native_dropout]
        buf13 = aten.native_dropout(reinterpret_tensor(buf12, (1, 1024, 1024), (1048576, 1024, 1), 0), 0.1, True)
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        buf19 = reinterpret_tensor(buf12, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf12  # reuse
        buf20 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        buf55 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_2, l__self___encoder_attn_q_proj, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_4.run(primals_27, buf14, primals_9, primals_10, buf19, buf20, buf55, 1024, 1024, grid=grid(1024), stream=stream0)
        buf21 = reinterpret_tensor(buf14, (1024, 1024), (1024, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf20, reinterpret_tensor(primals_11, (1024, 1024), (1, 1024), 0), out=buf21)
        buf22 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_29, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_13, (1024, 1024), (1, 1024), 0), out=buf22)
        buf23 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(primals_29, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_15, (1024, 1024), (1, 1024), 0), out=buf23)
        buf24 = empty((1, 16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf21, primals_12, buf24, 1048576, grid=grid(1048576), stream=stream0)
        del primals_12
        buf25 = reinterpret_tensor(buf21, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf21  # reuse
        # Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf23, primals_16, buf25, 1048576, grid=grid(1048576), stream=stream0)
        del primals_16
        buf26 = reinterpret_tensor(buf23, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf23  # reuse
        # Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf22, primals_14, buf26, 1048576, grid=grid(1048576), stream=stream0)
        del primals_14
        buf27 = buf6; del buf6  # reuse
        # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf26, (16, 64, 1024), (65536, 1, 64), 0), out=buf27)
        buf28 = empty((16, 1024, 1), device='cuda', dtype=torch.float32)
        buf29 = empty((16, 1024, 1), device='cuda', dtype=torch.float32)
        buf30 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf27, buf28, buf29, buf30, 16384, 1024, grid=grid(16384), stream=stream0)
        buf31 = reinterpret_tensor(buf22, (16, 1024, 64), (65536, 64, 1), 0); del buf22  # reuse
        # Source Nodes: [attn_output_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf30, reinterpret_tensor(buf25, (16, 1024, 64), (65536, 64, 1), 0), out=buf31)
        buf32 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_4], Original ATen: [aten.view]
        triton_poi_fused_view_3.run(buf31, buf32, 1048576, grid=grid(1048576), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (1024, 1024), (1024, 1), 0); del buf31  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_18, buf32, reinterpret_tensor(primals_17, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf33)
        del primals_18
        # Source Nodes: [hidden_states_5], Original ATen: [aten.native_dropout]
        buf34 = aten.native_dropout(reinterpret_tensor(buf33, (1, 1024, 1024), (1048576, 1024, 1), 0), 0.1, True)
        buf35 = buf34[0]
        buf36 = buf34[1]
        del buf34
        buf40 = reinterpret_tensor(buf33, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf33  # reuse
        buf41 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        buf54 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_6, l__self___fc1, residual_1, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf19, primals_9, primals_10, buf35, primals_19, primals_20, buf40, buf41, buf54, 1024, 1024, grid=grid(1024), stream=stream0)
        del primals_10
        buf42 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__self___fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_22, buf41, reinterpret_tensor(primals_21, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf42)
        del primals_22
        buf43 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_10, hidden_states_8], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf42, buf43, 4194304, grid=grid(4194304), stream=stream0)
        buf44 = reinterpret_tensor(buf35, (1024, 1024), (1024, 1), 0); del buf35  # reuse
        # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_24, buf43, reinterpret_tensor(primals_23, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf44)
        del primals_24
        # Source Nodes: [hidden_states_11], Original ATen: [aten.native_dropout]
        buf45 = aten.native_dropout(reinterpret_tensor(buf44, (1, 1024, 1024), (1048576, 1024, 1), 0), 0.1, True)
        buf46 = buf45[0]
        buf47 = buf45[1]
        del buf45
        buf51 = reinterpret_tensor(buf44, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf44  # reuse
        buf52 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf53 = empty((1, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_12, hidden_states_13, residual_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf40, primals_19, primals_20, buf46, primals_25, primals_26, buf51, buf52, buf53, 1024, 1024, grid=grid(1024), stream=stream0)
        del buf46
        del primals_20
        del primals_26
        return (buf52, primals_9, primals_19, primals_25, reinterpret_tensor(primals_27, (1024, 1024), (1024, 1), 0), buf11, buf15, buf19, buf20, reinterpret_tensor(primals_29, (1024, 1024), (1024, 1), 0), buf27, buf28, buf29, buf32, buf36, buf40, buf41, buf42, buf43, buf47, buf51, buf53, reinterpret_tensor(primals_23, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_21, (4096, 1024), (1024, 1), 0), buf54, reinterpret_tensor(primals_17, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf30, (16, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf25, (16, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf24, (16, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf26, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(primals_15, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_13, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_11, (1024, 1024), (1024, 1), 0), buf55, reinterpret_tensor(primals_7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf9, (16, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf4, (16, 64, 1024), (65536, 1, 64), 0), buf56, reinterpret_tensor(buf3, (16, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf5, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(primals_5, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_3, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_1, (1024, 1024), (1024, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForConditionalGeneration', benchmark_compiled_module)
