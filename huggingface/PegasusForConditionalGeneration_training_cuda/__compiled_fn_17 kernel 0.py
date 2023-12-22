
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


# kernel path: /tmp/torchinductor_youkaichao/ai/cai2fkaohpqsjcntc7yozd2akt3qqntj3ux6ip775z4cgsqx2dgn.py
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
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_0', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 128
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
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 1024.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/up/cupzcdhfdrfgfsd23hh4pgbludzqt6ykuibcz2opigz7tyscirra.py
# Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# contiguous_2 => clone_2
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlxovjarmsxrm5qzxs6mvk4l7uvxavse2u5cwtpzgmrafjic56m.py
# Source Nodes: [value_states], Original ATen: [aten.clone]
# value_states => clone_1
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5qjuqqpvwhwjthoxpxdztgn6bzwpj3iwj2aigexp4l4tmifvl7.py
# Source Nodes: [attn_weights_1], Original ATen: [aten._softmax]
# attn_weights_1 => amax, div, exp, sub_1, sum_1
triton_per_fused__softmax_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp11, rmask)
    tl.store(out_ptr0 + (x0), tmp4, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ta/ctabwgd4vktoovcve4oytd3rh27we4tpazzoh4jkjlxygcdgm6jg.py
# Source Nodes: [hidden_states_1], Original ATen: [aten.view]
# hidden_states_1 => view_14
triton_poi_fused_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjajhk7qm2sv6vffk6inspzc3edki64giwjrbk6y4qhe5vpzewr5.py
# Source Nodes: [hidden_states_4, l__self___fc1, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# hidden_states_4 => add_3, add_4, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
# l__self___fc1 => view_16
# residual_1 => add_2
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/5k/c5k5ateshahm3vy773ms2xfwey2otii3gjg7vjmiod7aq5tukptw.py
# Source Nodes: [hidden_states_5, hidden_states_7], Original ATen: [aten.gelu, aten.view]
# hidden_states_5 => add_5, erf, mul_5, mul_6, mul_7
# hidden_states_7 => view_18
triton_poi_fused_gelu_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /tmp/torchinductor_youkaichao/am/cam7h4upvoaaamnxjx27tfyhpolyc7dor4qvvgpxezhpzz7tysox.py
# Source Nodes: [hidden_states_9, residual_1], Original ATen: [aten.add]
# hidden_states_9 => add_6
# residual_1 => add_2
triton_poi_fused_add_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_3, (1024, 1024), (1024, 1))
    assert_size_stride(primals_4, (1024, ), (1, ))
    assert_size_stride(primals_5, (1024, 1024), (1024, 1))
    assert_size_stride(primals_6, (1024, ), (1, ))
    assert_size_stride(primals_7, (1024, 1024), (1024, 1))
    assert_size_stride(primals_8, (1024, ), (1, ))
    assert_size_stride(primals_9, (1024, 1024), (1024, 1))
    assert_size_stride(primals_10, (1024, ), (1, ))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_13, (4096, 1024), (1024, 1))
    assert_size_stride(primals_14, (4096, ), (1, ))
    assert_size_stride(primals_15, (1024, 4096), (4096, 1))
    assert_size_stride(primals_16, (1024, ), (1, ))
    assert_size_stride(primals_17, (1, 128, 1024), (131072, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf3 = reinterpret_tensor(buf1, (1, 128, 1), (128, 1, 1), 0); del buf1  # reuse
        buf4 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states, l__self___self_attn_q_proj], Original ATen: [aten.native_layer_norm, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_layer_norm_view_0.run(buf3, primals_17, primals_1, primals_2, buf0, buf4, 128, 1024, grid=grid(128), stream=stream0)
        del primals_2
        buf5 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(primals_3, (1024, 1024), (1, 1024), 0), out=buf5)
        buf6 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(primals_5, (1024, 1024), (1, 1024), 0), out=buf6)
        buf7 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(primals_7, (1024, 1024), (1, 1024), 0), out=buf7)
        buf8 = empty((1, 16, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf5, primals_4, buf8, 131072, grid=grid(131072), stream=stream0)
        del primals_4
        buf9 = reinterpret_tensor(buf5, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf5  # reuse
        # Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf7, primals_8, buf9, 131072, grid=grid(131072), stream=stream0)
        del primals_8
        buf10 = reinterpret_tensor(buf7, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf7  # reuse
        # Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf6, primals_6, buf10, 131072, grid=grid(131072), stream=stream0)
        del primals_6
        buf11 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf10, (16, 64, 128), (8192, 1, 64), 0), out=buf11)
        buf12 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf13 = empty((16, 128, 1), device='cuda', dtype=torch.float32)
        buf14 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf11, buf12, buf13, buf14, 2048, 128, grid=grid(2048), stream=stream0)
        buf15 = reinterpret_tensor(buf6, (16, 128, 64), (8192, 64, 1), 0); del buf6  # reuse
        # Source Nodes: [attn_output], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf14, reinterpret_tensor(buf9, (16, 128, 64), (8192, 64, 1), 0), out=buf15)
        buf16 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_1], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf15, buf16, 131072, grid=grid(131072), stream=stream0)
        buf17 = reinterpret_tensor(buf15, (128, 1024), (1024, 1), 0); del buf15  # reuse
        # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, buf16, reinterpret_tensor(primals_9, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf17)
        del primals_10
        # Source Nodes: [hidden_states_2], Original ATen: [aten.native_dropout]
        buf18 = aten.native_dropout(reinterpret_tensor(buf17, (1, 128, 1024), (131072, 1024, 1), 0), 0.1, True)
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf24 = reinterpret_tensor(buf17, (1, 128, 1024), (131072, 1024, 1), 0); del buf17  # reuse
        buf25 = empty((128, 1024), device='cuda', dtype=torch.float32)
        buf33 = empty((1, 128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_4, l__self___fc1, residual_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(primals_17, buf19, primals_11, primals_12, buf24, buf25, buf33, 128, 1024, grid=grid(128), stream=stream0)
        del primals_12
        buf26 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__self___fc1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, buf25, reinterpret_tensor(primals_13, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf26)
        del primals_14
        buf27 = empty((128, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_5, hidden_states_7], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_6.run(buf26, buf27, 524288, grid=grid(524288), stream=stream0)
        buf28 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_16, buf27, reinterpret_tensor(primals_15, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf28)
        del primals_16
        # Source Nodes: [hidden_states_8], Original ATen: [aten.native_dropout]
        buf29 = aten.native_dropout(reinterpret_tensor(buf28, (1, 128, 1024), (131072, 1024, 1), 0), 0.1, True)
        del buf28
        buf30 = buf29[0]
        buf31 = buf29[1]
        del buf29
        buf32 = buf30; del buf30  # reuse
        # Source Nodes: [hidden_states_9, residual_1], Original ATen: [aten.add]
        triton_poi_fused_add_7.run(buf32, primals_17, buf19, 131072, grid=grid(131072), stream=stream0)
        return (buf32, primals_1, primals_11, primals_17, buf0, buf3, buf4, buf11, buf12, buf13, buf16, buf20, buf24, buf25, buf26, buf27, buf31, reinterpret_tensor(primals_15, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_13, (4096, 1024), (1024, 1), 0), buf33, reinterpret_tensor(primals_9, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf14, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf9, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf8, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf10, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_5, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_3, (1024, 1024), (1024, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PegasusForConditionalGeneration', benchmark_compiled_module)
