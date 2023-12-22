
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


# kernel path: /tmp/torchinductor_youkaichao/ev/cevxkoylo5czbwdmc4dkuxfnef5josq5mzhbbfk46xfomgwuw7dl.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (131072*r2)), rmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = 64.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr2 + (x0), tmp19, None)
    tl.store(out_ptr0 + (x0), tmp9, None)
    tl.store(out_ptr1 + (x0), tmp17, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/jl/cjl7c4luh7boog7rch5helk7cxuugwiiyyrv3zevka2bnqkyo5nl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = 64.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.001953125
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3wtxz6w7mkaassd5a5h3glnbfmg5rajnlkcsrjr7ppypufwq5o.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 512
    x2 = (xindex // 1024)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (64*x0) + (32768*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (64*x4)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chk4sk6huy7loywraabick3vxtfoxcxc7dxcpgnf2p6t4y5fxjlq.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_poi_fused__softmax_backward_data_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp2 - tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktzlbzlyrpa4ab7pujq7prikdcb2sjet4ylcamcfxfuofko5zjt.py
# Source Nodes: [x_gap_163], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# x_gap_163 => var_mean_137
triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp25, xmask)
    tl.store(out_ptr2 + (x0), tmp31, xmask)
    tl.store(out_ptr3 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tu/ctu5i3eiqlw3ratypeg4fniyctzwg4yiy25teb3qftcxf7ln3b5w.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55upe2utnmgwisctjqivs4kzjpr4er6fwu4vbevqnhz52egomi5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (65536*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*(x0 % 512)) + (32768*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + ((512*r2) + (x0 % 512)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1 + (64*x0) + (65536*r2)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp15 - tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr2 + (x0), tmp24, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvfuusa7mhsx3okrlwqg3rpdhwijapj2crqkhhi52qwl3n3dnop.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x4 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (64*(x1 % 512)) + (32768*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((512*x2) + (x1 % 512)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.001953125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3kpni7c6hfc2unawsufvkewdalxd62f5g3bdwetyv2er3ijhoj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pz/cpzg7qy7ytt6zrld32ddykdh777s7p4sunttqb5dzqvef3p2evhf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.001953125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlofwc2hey5rfb2fbxnla6z3uquai4tgyfwv7mo343dfyufdnnj.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (131072*r2)), rmask).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp15 - tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr2 + (x0), tmp24, None)
    tl.store(out_ptr0 + (x0), tmp14, None)
    tl.store(out_ptr1 + (x0), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/carz542cxkqjp476nv5cw4hkuj27cx2nzbseorzbw733w4pys6ri.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x3), None)
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.001953125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crx3uc4hgvlh3coxpjrxfey3ezeo3j5onc7mxkiqz7x4i424cmei.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]

triton_poi_fused_add_div_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None).to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr0 + (x2), None)
    tmp13 = tl.load(in_ptr4 + (x2), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/conptnsihgrap4s63yuyk6po2252evzk6nv5sb4cnnuqpyzql4jo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 - tmp14
    tmp16 = tmp0 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp12 * tmp21
    tmp24 = tmp20 * tmp23
    tl.store(out_ptr3 + (x0), tmp22, None)
    tl.store(out_ptr4 + (x0), tmp24, None)
    tl.store(out_ptr0 + (x0), tmp4, None)
    tl.store(out_ptr1 + (x0), tmp12, None)
    tl.store(out_ptr2 + (x0), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfyb5vadyyovvw6mn7liriingima475kyelhzljmnt2wwtvati6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x3), None)
    tmp19 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.001953125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexdzqymc5smfx2hwolibn75bakce3ujyo564dcpoawmc6sbmaqh.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + ((1 + x1) // 2)))))) + (8*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + ((1 + x1) // 2))))) >= 0, 0, 8))) + (64*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + ((1 + x0) // 2))))) >= 0, 0, 8))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + ((1 + x1) // 2)))))) + (8*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + ((1 + x1) // 2))))) >= 0, 0, 8))) + (64*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(8, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(8, 1 + ((1 + x0) // 2))))) >= 0, 0, 8))), None)
    tmp18 = tl.load(in_ptr0 + ((8*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(8, 1 + ((1 + x1) // 2)))))) + (8*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(8, 1 + ((1 + x1) // 2))))) >= 0, 0, 8))) + (64*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + ((1 + x0) // 2))))) >= 0, 0, 8))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + ((8*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(8, 1 + ((1 + x1) // 2)))))) + (8*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(8, 1 + ((1 + x1) // 2))))) >= 0, 0, 8))) + (64*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(8, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(8, 1 + ((1 + x0) // 2))))) >= 0, 0, 8))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(8, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(8, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnvba7irjail7wqasjnqszjyos4lbqg5b6uz4t2srx6u6d4cp32.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x2 = (xindex // 1024)
    x4 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (256*x0) + (131072*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vo/cvoig2ofpqaaeqp4trok7f5ccb3gy6seumihcd5uudrqrmuiztuc.py
# Source Nodes: [x_gap_153], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# x_gap_153 => var_mean_128
triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnw5lkoqb5kdztfdp4l624zgzcr2xciiwc4an5oqgmulocw3jc3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (256*(x0 % 512)) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((512*r2) + (x0 % 512)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 256.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp19, xmask)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr2 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5tl57dgy25lxaqq47pccjyqityzsb2dddt6xybiqhzsxwmse42.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256) % 1024
    x2 = (xindex // 262144)
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (256*(x1 % 512)) + (131072*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((512*x2) + (x1 % 512)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00048828125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvc6cbspk67xdf575fbyaz6issrb2l74wrqj7gx6osmwbjus4kvi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nr/cnrk5lj5yiagxgy4wvvehsmycpyevnimpmdheasblqpudc4cxkv5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfupw43mhcddnaqlcvt5rqnkcuxkqlcspu4dvvtioz3mez6q7xjx.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 256)
        r4 = rindex % 256
        r1 = rindex % 16
        r2 = (rindex // 16) % 16
        tmp0 = tl.load(in_ptr0 + (r4 + (256*x0) + (262144*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((8*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(8, 1 + (r2 // 2)))))) + (8*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(8, 1 + (r2 // 2))))) >= 0, 0, 8))) + (64*x0) + (65536*r3) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(8, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(8, 1 + (r1 // 2))))) >= 0, 0, 8))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (r4 + (256*x0) + (262144*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r4 + (256*x0) + (262144*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 / 4
        tmp5 = tl.math.max(0, (r2 // 2))
        tmp6 = tl.math.min(8, 1 + (r2 // 2))
        tmp7 = tmp5 < tmp6
        tmp8 = tl.math.max(0, (r1 // 2))
        tmp9 = tl.math.min(8, 1 + (r1 // 2))
        tmp10 = tmp8 < tmp9
        tmp11 = tmp7 & tmp10
        tmp12 = tl.where(tmp11, tmp4, tmp1)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(tmp2, tmp1, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp24, xmask)
    tmp26 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr2 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z6/cz6xzx52oxz3pejm3l4iasc2jjlmphfeknsnuvtbdopiqt6ctdks.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x5 = (xindex // 256)
    x2 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + ((8*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + (x1 // 2)))))) + (8*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + (x1 // 2))))) >= 0, 0, 8))) + (64*x5) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + (x0 // 2))))) >= 0, 0, 8))), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x4), None)
    tmp16 = tl.load(in_ptr3 + (x4), None)
    tmp17 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (x1 // 2))
    tmp6 = tl.math.min(8, 1 + (x1 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (x0 // 2))
    tmp9 = tl.math.min(8, 1 + (x0 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp18 = tmp16 - tmp17
    tmp20 = 0.00048828125
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(in_out_ptr0 + (x4), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crx3sv2b32ifj2ju3xkh4nl56zonris6bm3j3arwcedac4rv2nnw.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 256
    x2 = (xindex // 512)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (256*x0) + (65536*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaimk6nwr2lfprjjsd7lakopnnsfnkt4pr4xzy3xugynk6x7wlx.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_poi_fused__softmax_backward_data_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (256 + x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (256 + x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp2 - tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqckqkrxjnmejdlz3gt3irfibs43gnbkfgslmmstqvaxznbo3ca2.py
# Source Nodes: [x_gap_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# x_gap_148 => var_mean_124
triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czaukfnyqgtadkqo3onkiotb4bzlktzpinyktw5xjzg4m6yeoz4k.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cuxe2j57m5q2yyeugkow4e7d42u7q5atg2h4rsx53ltdigm3zfdx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (256*(x0 % 256)) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((256*r2) + (x0 % 256)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 256.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp19, xmask)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr2 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wv/cwvh4u25kbryy4vgll5i73voi2cncihmjchddfbavt5o5sdbpoir.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256) % 512
    x2 = (xindex // 131072)
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (256*(x1 % 256)) + (65536*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((256*x2) + (x1 % 256)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00048828125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/covjq5klmurrpplxf7c7kjvmcvwle3mph2mbhmnco7noe43t4ogx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxx5aechmq55qs4rmpzvpjuu4ozhgq25zzjtuq5p6rzuwjbujpr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kf/ckflg4o3r7vldbslcjx5mthugufiuefvrvhlxynccefgi3gijixj.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_threshold_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + ((8*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + (x1 // 2)))))) + (8*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + (x1 // 2))))) >= 0, 0, 8))) + (64*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + (x0 // 2))))) >= 0, 0, 8))), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp18 = tl.load(in_ptr3 + (x3), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tmp5 / 4
    tmp7 = tl.math.max(0, (x1 // 2))
    tmp8 = tl.math.min(8, 1 + (x1 // 2))
    tmp9 = tmp7 < tmp8
    tmp10 = tl.math.max(0, (x0 // 2))
    tmp11 = tl.math.min(8, 1 + (x0 // 2))
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tl.where(tmp13, tmp6, tmp1)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(tmp4, tmp1, tmp16)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.where(tmp2, tmp1, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn35hubnqfuqgmbqlvynhyfdamnmofxsit6uzewmfehttuggv5ui.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck443tc3tt4rfypniudha23otvnvvypm6unz3bqfzurwvzkszb3e.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4lduvdxv52kmcrluc6e3hspctczvumcgz5k7dxhvkouxrgugxl.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zkp4annlrbju44pqabh75hccda436lerntzyknh4lfgirgp4kv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x3), None)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00048828125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl42rhq7nts4ae62zp3fliyox6ucsjpp6y3eypnf7xjx3jgnhjxc.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/csspdxerssltjskgul54bytinahjczfjfyv6y7taurp3gp3xboei.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp22, xmask)
    tmp24 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tmp15 * tmp24
    tmp27 = tmp22 * tmp26
    tl.store(out_ptr3 + (x0), tmp25, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jd/cjd2nredlslhoyug75vfrws2czlsamxhakauil6jempiwjrgochm.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x3), None)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x3), None)
    tmp25 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00048828125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x3), tmp23, None)
    tl.store(out_ptr1 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cog5jw7ok6h235t6f5zzusf6g5fnw77jmrwujmyribgtkfqy35jp.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x2 = (xindex // 1024)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((16*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(16, 1 + ((1 + x1) // 2)))))) + (16*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(16, 1 + ((1 + x1) // 2))))) >= 0, 0, 16))) + (256*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(16, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(16, 1 + ((1 + x0) // 2))))) >= 0, 0, 16))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((16*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(16, 1 + ((1 + x1) // 2)))))) + (16*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(16, 1 + ((1 + x1) // 2))))) >= 0, 0, 16))) + (256*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(16, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(16, 1 + ((1 + x0) // 2))))) >= 0, 0, 16))), None)
    tmp18 = tl.load(in_ptr0 + ((16*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(16, 1 + ((1 + x1) // 2)))))) + (16*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(16, 1 + ((1 + x1) // 2))))) >= 0, 0, 16))) + (256*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(16, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(16, 1 + ((1 + x0) // 2))))) >= 0, 0, 16))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + ((16*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(16, 1 + ((1 + x1) // 2)))))) + (16*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(16, 1 + ((1 + x1) // 2))))) >= 0, 0, 16))) + (256*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(16, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(16, 1 + ((1 + x0) // 2))))) >= 0, 0, 16))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(16, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(16, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gy/cgy5tngvlj5j2hxr3dpnttrw34mwcooxamqcmcm7gun6si7hcnuh.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 256
    x2 = (xindex // 512)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (1024*x0) + (262144*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (1024*x4)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7t5vh4pbvsanbgjcjgnccnd7qh7ivvz4gaet3qamihna5aztzkm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*(x0 % 256)) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((256*r2) + (x0 % 256)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 1024.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp19, xmask)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr2 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkw35zoluv4waa2olt5vct6odpu3k6up46frjxqes2vtn2kqjg7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 512
    x2 = (xindex // 524288)
    x4 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (1024*(x1 % 256)) + (262144*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((256*x2) + (x1 % 256)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.0001220703125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dmbyb6xnhirgpsciwrvqimpppkpnxu2y335iizrss6jpxqv7wy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civqaljpcqtk7s7tpbjarffftxyckbcai4ees77qtpxzddjvj5ay.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xatzelzeeaohkmomyfj4busu6mmoibgpmbw4mzfg7vwtzfqf3b.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 1024)
        r4 = rindex % 1024
        r1 = rindex % 32
        r2 = (rindex // 32) % 32
        tmp0 = tl.load(in_ptr0 + (r4 + (1024*x0) + (524288*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(16, 1 + (r2 // 2)))))) + (16*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(16, 1 + (r2 // 2))))) >= 0, 0, 16))) + (256*x0) + (131072*r3) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(16, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(16, 1 + (r1 // 2))))) >= 0, 0, 16))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (r4 + (1024*x0) + (524288*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r4 + (1024*x0) + (524288*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 / 4
        tmp5 = tl.math.max(0, (r2 // 2))
        tmp6 = tl.math.min(16, 1 + (r2 // 2))
        tmp7 = tmp5 < tmp6
        tmp8 = tl.math.max(0, (r1 // 2))
        tmp9 = tl.math.min(16, 1 + (r1 // 2))
        tmp10 = tmp8 < tmp9
        tmp11 = tmp7 & tmp10
        tmp12 = tl.where(tmp11, tmp4, tmp1)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(tmp2, tmp1, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp24, xmask)
    tmp26 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr2 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyiaz7albmwabtvvohljp4kogcpr2qfdq3tszjx7rg7dsageylma.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x5 = (xindex // 1024)
    x2 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + ((16*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(16, 1 + (x1 // 2)))))) + (16*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(16, 1 + (x1 // 2))))) >= 0, 0, 16))) + (256*x5) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(16, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(16, 1 + (x0 // 2))))) >= 0, 0, 16))), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x4), None)
    tmp16 = tl.load(in_ptr3 + (x4), None)
    tmp17 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (x1 // 2))
    tmp6 = tl.math.min(16, 1 + (x1 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (x0 // 2))
    tmp9 = tl.math.min(16, 1 + (x0 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0001220703125
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(in_out_ptr0 + (x4), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/coke2nygsunnzbarzfp3ik7satj66ao4uflgbskapjgxyhfgwutf.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 128
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (1024*x0) + (131072*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (1024*x4)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnopvkm2i5ocu4r4yka54zckdfs7vzhnvhtgdmvvq4zkgdthbnk.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_poi_fused__softmax_backward_data_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (128 + x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (128 + x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp2 - tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3dwugkbbauwfh7ewrtgnvzjrl45pfjpuklavooaktvv4mq3pm3.py
# Source Nodes: [x_gap_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# x_gap_33 => var_mean_31
triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rvoa2eoet6udwwtkuaxmp24ygzo3r2fxvn273qnbeimte277qc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3qfzkj36ob4i7yzrb72k5kyr2wif6fm3p4phmwcjzxmr7gpmuk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*(x0 % 128)) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((128*r2) + (x0 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 1024.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp19, xmask)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr2 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca753eyoqidskjk5sf27qm6aitxat7vzonyq464esjqkwyzciw36.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 256
    x2 = (xindex // 262144)
    x4 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (1024*(x1 % 128)) + (131072*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((128*x2) + (x1 % 128)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.0001220703125
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/cht6h7jpcmqb24hmcv53zqx7ymu5uwkgfgsbo3wu6so4ahu76ft6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/cow2wwznzqgi76hsfvkbn7krptstksmdnkluf44vibayrekmj2lj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzpaupbeutb2srwg5s4oszt5vccbpzakojjwvsibc54xgfiucds.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x2 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + ((16*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(16, 1 + (x1 // 2)))))) + (16*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(16, 1 + (x1 // 2))))) >= 0, 0, 16))) + (256*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(16, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(16, 1 + (x0 // 2))))) >= 0, 0, 16))), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp18 = tl.load(in_ptr3 + (x3), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tmp5 / 4
    tmp7 = tl.math.max(0, (x1 // 2))
    tmp8 = tl.math.min(16, 1 + (x1 // 2))
    tmp9 = tmp7 < tmp8
    tmp10 = tl.math.max(0, (x0 // 2))
    tmp11 = tl.math.min(16, 1 + (x0 // 2))
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tl.where(tmp13, tmp6, tmp1)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(tmp4, tmp1, tmp16)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.where(tmp2, tmp1, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cne2s7vwp2nhodzujdghyqlpwgvzuaskptflgt4ygaw7lj5ny2y7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zc/czcvjpengj5jyq2zdlcmk32kp3ugkoeqgdz2tm5v5vo5qjmx4pk6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0001220703125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw44r5rnirh76da6zorbhdr5prokelwpfgp2lyrq7ngu3ho2c6zj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hw/chwfjzbwgeax43dzr7fhodjp6ofc3eqky3xc3fw4semfigdyddmi.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x3), None)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0001220703125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32mzsau5pdekvopsutcvkwmivupmtt6333ix5fmuslpj5s6aop2.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlezz4j74hhwp3pl7ddf2fdnebkboetah5tmkxfvmpr47cuofno.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp0 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tmp9 * tmp18
    tmp21 = tmp16 * tmp20
    tl.store(out_ptr3 + (x0), tmp19, xmask)
    tl.store(out_ptr4 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv27agu6nagvtzdpsrjyr5cfgsougjilzitxtsuu3ide33ejcaeq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x3), None)
    tmp19 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0001220703125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curaywmjyqrljz6s4ck47vd5zuqxqgp4pcco6q67uwt3hprtquie.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x2 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + ((1 + x1) // 2)))))) + (32*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + ((1 + x1) // 2))))) >= 0, 0, 32))) + (1024*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + ((1 + x0) // 2))))) >= 0, 0, 32))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((32*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + ((1 + x1) // 2)))))) + (32*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + ((1 + x1) // 2))))) >= 0, 0, 32))) + (1024*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(32, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(32, 1 + ((1 + x0) // 2))))) >= 0, 0, 32))), None)
    tmp18 = tl.load(in_ptr0 + ((32*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(32, 1 + ((1 + x1) // 2)))))) + (32*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(32, 1 + ((1 + x1) // 2))))) >= 0, 0, 32))) + (1024*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + ((1 + x0) // 2))))) >= 0, 0, 32))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + ((32*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(32, 1 + ((1 + x1) // 2)))))) + (32*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(32, 1 + ((1 + x1) // 2))))) >= 0, 0, 32))) + (1024*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(32, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(32, 1 + ((1 + x0) // 2))))) >= 0, 0, 32))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(32, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(32, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5cczv2kkf5wh3s5e2ar5o4h4xstof7pvcuruzvxuie5hmjsmtb.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x2 = (xindex // 256)
    x4 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (4096*x0) + (524288*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (4096*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckbftxs3pql3wn2dhgkn7wcgd743zvbzmghwaorpksfqh63d5w2y.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (4096*(x0 % 128)) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((128*r2) + (x0 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 4096.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp19, xmask)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr2 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7ltbakrssdykq43cdgfo7kolqzavsm5v5tarkdaph6f7f2zazq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    x1 = (xindex // 4096) % 256
    x2 = (xindex // 1048576)
    x4 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (4096*(x1 % 128)) + (524288*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((128*x2) + (x1 % 128)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 4096.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 3.0517578125e-05
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zz/czzkixvaszfa3qfnmjfkesqm3wlssykxgicxw3k72uulvpy3xobi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3ksegsibg2m4irx3z6g6n52yp3rceja44avgemv4bd5i266s3o.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crxfdwdnopcgltavs6jnsk2dtzow4albfnprisam6nqepz4h656v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/canz46rlx7wv7jyisvytgal7lzfsf72clvewqoxe5t2swa2d3eci.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.0517578125e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlmod4vcmt7tscuckkbuehdeqiyps3key77dairhzgwm5cxgi5y.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 4096)
        r4 = rindex % 4096
        r1 = rindex % 64
        r2 = (rindex // 64) % 64
        tmp0 = tl.load(in_ptr0 + (r4 + (4096*x0) + (1048576*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(32, 1 + (r2 // 2)))))) + (32*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(32, 1 + (r2 // 2))))) >= 0, 0, 32))) + (1024*x0) + (262144*r3) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(32, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(32, 1 + (r1 // 2))))) >= 0, 0, 32))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (r4 + (4096*x0) + (1048576*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r4 + (4096*x0) + (1048576*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 / 4
        tmp5 = tl.math.max(0, (r2 // 2))
        tmp6 = tl.math.min(32, 1 + (r2 // 2))
        tmp7 = tmp5 < tmp6
        tmp8 = tl.math.max(0, (r1 // 2))
        tmp9 = tl.math.min(32, 1 + (r1 // 2))
        tmp10 = tmp8 < tmp9
        tmp11 = tmp7 & tmp10
        tmp12 = tl.where(tmp11, tmp4, tmp1)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.where(tmp2, tmp1, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp24, xmask)
    tmp26 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr2 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7muhpwqv5rwjlbg66dmlx5dkzwihustddd62m5zzu73hjhcietu.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x5 = (xindex // 4096)
    x2 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + ((32*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + (x1 // 2)))))) + (32*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + (x1 // 2))))) >= 0, 0, 32))) + (1024*x5) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + (x0 // 2))))) >= 0, 0, 32))), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x4), None)
    tmp16 = tl.load(in_ptr3 + (x4), None)
    tmp17 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (x1 // 2))
    tmp6 = tl.math.min(32, 1 + (x1 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (x0 // 2))
    tmp9 = tl.math.min(32, 1 + (x0 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp18 = tmp16 - tmp17
    tmp20 = 3.0517578125e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(in_out_ptr0 + (x4), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/ctthnoxlkogfyxcrfugazaqwobe442q6j5cedjinbpvieol2a7a6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x2 = (xindex // 128)
    x4 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (4096*x0) + (262144*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (4096*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyxcaozdcu2fnmdtdpis75usjtyuaabv7c2ekyhpb3msrawtwi2.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_poi_fused__softmax_backward_data_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (64 + x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (64 + x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp2 - tmp10
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqv2fo74wvwz7izyrvs4uzyrczotfm3x2yomeoc4kztp3ssdcyf.py
# Source Nodes: [x_gap_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
# x_gap_13 => var_mean_14
triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/52/c52g2erxfyhi2daufrg2my25kt4vf5rl7jz3vk4f4lyijrobf5oj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd22vd3bbtbaelxf4aagaofspkukjh74exm6xqtipecemdyoezqg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((4096*(x0 % 64)) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*(r2 // 4096)) + (256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((64*(r2 // 4096)) + (128*x1) + (x0 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 4096.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cuxbnnzpbeoshxgeitdxwbyjyai36uag22iswwqj7tdjbihxai4w.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 4096
    x1 = (xindex // 4096) % 128
    x2 = (xindex // 524288)
    x4 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (4096*(x1 % 64)) + (262144*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + ((64*x2) + (x1 % 64)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x3), None)
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 4096.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 3.0517578125e-05
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ad/cadkvxraw4kiydglnrkhmahcbbdpy6vspfanggylqcahls2jpb26.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnf7ogj7lzvixq64poh4fvtihiggy7l4qewnl3dbvqyljltb66j.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbbul4xkxhegcnfl6nlqlaeyarub6ifwq2wrihzcktutb4r36ay.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2instgdvtpmwraaywjwqqbov5splpbimlfy75oyshmk56wztfoa.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.0517578125e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/cam4pcu35szplhycnbmvugnme6cyzoma37utkfgp2ix64ziorxkc.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.threshold_backward]

triton_poi_fused_add_avg_pool2d_backward_threshold_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_threshold_backward_84', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x2 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + ((32*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + (x1 // 2)))))) + (32*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(32, 1 + (x1 // 2))))) >= 0, 0, 32))) + (1024*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + (x0 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(32, 1 + (x0 // 2))))) >= 0, 0, 32))), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp18 = tl.load(in_ptr3 + (x3), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tmp5 / 4
    tmp7 = tl.math.max(0, (x1 // 2))
    tmp8 = tl.math.min(32, 1 + (x1 // 2))
    tmp9 = tmp7 < tmp8
    tmp10 = tl.math.max(0, (x0 // 2))
    tmp11 = tl.math.min(32, 1 + (x0 // 2))
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tl.where(tmp13, tmp6, tmp1)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.where(tmp4, tmp1, tmp16)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.where(tmp2, tmp1, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdncwqarqfh3ukfzncoohqro2ukek63gupjbbh2noyzglzlw7swe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mls6rgymsjuhiktkvieutgrkabmdk6ae3s5zcyd3r3ppysm3eu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 3.0517578125e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffwcnpe6hbtmdkkfq6gjlh4jhdai3tciqnzqmevdu7ccdfhfnzn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp22, xmask)
    tmp24 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tmp15 * tmp24
    tmp27 = tmp22 * tmp26
    tl.store(out_ptr3 + (x0), tmp25, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ny/cnynxigygxlsi37lecbuefwfk4rfjoxqjxt2vwjftfubpbvlhouw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x3), None)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x3), None)
    tmp25 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.0517578125e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x3), tmp23, None)
    tl.store(out_ptr1 + (x3), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3ogyoxar3ylsbzkkmxxlhrm7gdzyv26c43qimayr3i2gziodmk.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5hokjerf7sryymyyf6pz2cjoycl32ccmigwmwvu55l32tp5kkw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6fimrhsoz27gydpplqdz535l3rveof4ack45qkrlxanyamm7mu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d6/cd6mw357prouzaqcrfw2675rciyhl3rggfisj4cntaioaf5xh7l5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4cqvpxd2rd7v4upqmnd2lds7uhh3cqrzilxcsxotkks5gkqh4mh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnjadm7m67xoyyp7vpvtpnhtedhyqdoceze5glf2sbofnejwd5b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlgei2j546mo23oznstu3cexe3xs7d2u6bti563tckjv6x6koqm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqdagjynhdu7huscvz35uqnuxh34gmwb4c2wju6f2djv32cc27i.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_96', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3sqhmg6lc2ziqeopbfc7ezvkliz2zvb7k4swokam7lm4kemado.py
# Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]

triton_poi_fused_add_max_pool2d_with_indices_backward_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_backward_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 128
    x2 = (xindex // 16384)
    x3 = xindex % 16384
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2)))))) + (64*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2))))) >= 0, 0, 64))) + (4096*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) >= 0, 0, 64))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((64*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2)))))) + (64*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2))))) >= 0, 0, 64))) + (4096*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) >= 0, 0, 64))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((64*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2)))))) + (64*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2))))) >= 0, 0, 64))) + (4096*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) >= 0, 0, 64))), None)
    tmp7 = tl.load(in_ptr1 + ((64*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2)))))) + (64*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2))))) >= 0, 0, 64))) + (4096*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) >= 0, 0, 64))), None)
    tmp19 = tl.load(in_ptr0 + ((64*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2)))))) + (64*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2))))) >= 0, 0, 64))) + (4096*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) >= 0, 0, 64))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((64*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2)))))) + (64*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2))))) >= 0, 0, 64))) + (4096*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) >= 0, 0, 64))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((64*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2)))))) + (64*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2))))) >= 0, 0, 64))) + (4096*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) >= 0, 0, 64))), None)
    tmp31 = tl.load(in_ptr1 + ((64*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2)))))) + (64*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x1) // 2))))) >= 0, 0, 64))) + (4096*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(64, 1 + ((1 + x0) // 2))))) >= 0, 0, 64))), None)
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (x1 // 2))
    tmp10 = tl.math.min(64, 1 + ((1 + x1) // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp13 = tl.math.min(64, 1 + ((1 + x0) // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.math.max(0, (x0 // 2))
    tmp25 = tmp24 < tmp13
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp21
    tmp28 = tmp18 + tmp20
    tmp29 = tl.where(tmp27, tmp28, tmp18)
    tmp32 = tmp30 == tmp2
    tmp33 = tmp23 & tmp14
    tmp34 = tmp33 & tmp32
    tmp35 = tmp29 + tmp31
    tmp36 = tl.where(tmp34, tmp35, tmp29)
    tl.store(out_ptr0 + (x5), tmp36, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvfqivqxh5bmnzlq5jbqgcrnzm6idsgajjj4iti3fcss2zuyhmz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nr/cnrqpcnzmc744j3ezpoxjhzb2axdsgso2qzwvdkii7iyluqiz7ye.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 7.62939453125e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbalx2op7h7g6aabmropexhnx2fppsupqq673y4hdzkhy3kfxk6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_100 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (1048576*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (1048576*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (1048576*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca76e3eycx7lsdfe55qvy5o3skqkei4msy6xpiwsqqd2obwa75r4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pt/cptd6thzk3bi2no2tjb7tvhypsljpcdwmg6m6je65z2dvvxsz6si.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5x45ftxtqcof6hyjx6rj76r5wbpw5r2tciumgqk2mf2hfem3fj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 7.62939453125e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_51, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_66, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_99, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_114, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_129, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_147, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_162, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_177, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_192, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_207, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_222, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_237, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_252, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_267, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_282, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_297, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_312, primals_314, primals_316, primals_317, primals_319, primals_320, primals_322, primals_323, primals_325, primals_327, primals_329, primals_331, primals_332, primals_334, primals_335, primals_337, primals_338, primals_340, primals_342, primals_344, primals_346, primals_347, primals_349, primals_350, primals_352, primals_353, primals_355, primals_357, primals_359, primals_361, primals_362, primals_364, primals_365, primals_367, primals_368, primals_370, primals_372, primals_374, primals_376, primals_377, primals_379, primals_380, primals_382, primals_383, primals_385, primals_387, primals_389, primals_391, primals_392, primals_394, primals_395, primals_397, primals_398, primals_400, primals_402, primals_404, primals_406, primals_407, primals_409, primals_410, primals_412, primals_413, primals_415, primals_417, primals_419, primals_421, primals_422, primals_424, primals_425, primals_427, primals_428, primals_430, primals_432, primals_434, primals_436, primals_437, primals_439, primals_440, primals_442, primals_443, primals_445, primals_447, primals_449, primals_451, primals_452, primals_454, primals_455, primals_457, primals_458, primals_460, primals_462, primals_464, primals_466, primals_467, primals_469, primals_470, primals_472, primals_473, primals_475, primals_477, primals_479, primals_481, primals_482, primals_484, primals_485, primals_487, primals_488, primals_490, primals_491, primals_493, primals_495, primals_497, primals_499, primals_500, primals_502, primals_503, primals_505, primals_506, primals_508, primals_510, primals_512, primals_514, primals_515, primals_936, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, mean, convolution_5, relu_5, div, sum_3, convolution_7, squeeze_19, convolution_8, squeeze_22, relu_6, convolution_9, squeeze_25, relu_7, convolution_10, squeeze_28, relu_8, mean_1, convolution_11, relu_9, div_1, sum_6, convolution_13, squeeze_34, relu_10, convolution_14, squeeze_37, relu_11, convolution_15, squeeze_40, relu_12, mean_2, convolution_16, relu_13, div_2, sum_9, convolution_18, squeeze_46, relu_14, convolution_19, squeeze_49, relu_15, convolution_20, squeeze_52, relu_16, mean_3, convolution_21, relu_17, div_3, sum_12, avg_pool2d, convolution_23, squeeze_58, avg_pool2d_1, convolution_24, squeeze_61, relu_18, convolution_25, squeeze_64, relu_19, convolution_26, squeeze_67, relu_20, mean_4, convolution_27, relu_21, div_4, sum_15, convolution_29, squeeze_73, relu_22, convolution_30, squeeze_76, relu_23, convolution_31, squeeze_79, relu_24, mean_5, convolution_32, relu_25, div_5, sum_18, convolution_34, squeeze_85, relu_26, convolution_35, squeeze_88, relu_27, convolution_36, squeeze_91, relu_28, mean_6, convolution_37, relu_29, div_6, sum_21, convolution_39, squeeze_97, relu_30, convolution_40, squeeze_100, relu_31, convolution_41, squeeze_103, relu_32, mean_7, convolution_42, relu_33, div_7, sum_24, avg_pool2d_2, convolution_44, squeeze_109, avg_pool2d_3, convolution_45, squeeze_112, relu_34, convolution_46, squeeze_115, relu_35, convolution_47, squeeze_118, relu_36, mean_8, convolution_48, relu_37, div_8, sum_27, convolution_50, squeeze_124, relu_38, convolution_51, squeeze_127, relu_39, convolution_52, squeeze_130, relu_40, mean_9, convolution_53, relu_41, div_9, sum_30, convolution_55, squeeze_136, relu_42, convolution_56, squeeze_139, relu_43, convolution_57, squeeze_142, relu_44, mean_10, convolution_58, relu_45, div_10, sum_33, convolution_60, squeeze_148, relu_46, convolution_61, squeeze_151, relu_47, convolution_62, squeeze_154, relu_48, mean_11, convolution_63, relu_49, div_11, sum_36, convolution_65, squeeze_160, relu_50, convolution_66, squeeze_163, relu_51, convolution_67, squeeze_166, relu_52, mean_12, convolution_68, relu_53, div_12, sum_39, convolution_70, squeeze_172, relu_54, convolution_71, squeeze_175, relu_55, convolution_72, squeeze_178, relu_56, mean_13, convolution_73, relu_57, div_13, sum_42, convolution_75, squeeze_184, relu_58, convolution_76, squeeze_187, relu_59, convolution_77, squeeze_190, relu_60, mean_14, convolution_78, relu_61, div_14, sum_45, convolution_80, squeeze_196, relu_62, convolution_81, squeeze_199, relu_63, convolution_82, squeeze_202, relu_64, mean_15, convolution_83, relu_65, div_15, sum_48, convolution_85, squeeze_208, relu_66, convolution_86, squeeze_211, relu_67, convolution_87, squeeze_214, relu_68, mean_16, convolution_88, relu_69, div_16, sum_51, convolution_90, squeeze_220, relu_70, convolution_91, squeeze_223, relu_71, convolution_92, squeeze_226, relu_72, mean_17, convolution_93, relu_73, div_17, sum_54, convolution_95, squeeze_232, relu_74, convolution_96, squeeze_235, relu_75, convolution_97, squeeze_238, relu_76, mean_18, convolution_98, relu_77, div_18, sum_57, convolution_100, squeeze_244, relu_78, convolution_101, squeeze_247, relu_79, convolution_102, squeeze_250, relu_80, mean_19, convolution_103, relu_81, div_19, sum_60, convolution_105, squeeze_256, relu_82, convolution_106, squeeze_259, relu_83, convolution_107, squeeze_262, relu_84, mean_20, convolution_108, relu_85, div_20, sum_63, convolution_110, squeeze_268, relu_86, convolution_111, squeeze_271, relu_87, convolution_112, squeeze_274, relu_88, mean_21, convolution_113, relu_89, div_21, sum_66, convolution_115, squeeze_280, relu_90, convolution_116, squeeze_283, relu_91, convolution_117, squeeze_286, relu_92, mean_22, convolution_118, relu_93, div_22, sum_69, convolution_120, squeeze_292, relu_94, convolution_121, squeeze_295, relu_95, convolution_122, squeeze_298, relu_96, mean_23, convolution_123, relu_97, div_23, sum_72, convolution_125, squeeze_304, relu_98, convolution_126, squeeze_307, relu_99, convolution_127, squeeze_310, relu_100, mean_24, convolution_128, relu_101, div_24, sum_75, convolution_130, squeeze_316, relu_102, convolution_131, squeeze_319, relu_103, convolution_132, squeeze_322, relu_104, mean_25, convolution_133, relu_105, div_25, sum_78, convolution_135, squeeze_328, relu_106, convolution_136, squeeze_331, relu_107, convolution_137, squeeze_334, relu_108, mean_26, convolution_138, relu_109, div_26, sum_81, convolution_140, squeeze_340, relu_110, convolution_141, squeeze_343, relu_111, convolution_142, squeeze_346, relu_112, mean_27, convolution_143, relu_113, div_27, sum_84, convolution_145, squeeze_352, relu_114, convolution_146, squeeze_355, relu_115, convolution_147, squeeze_358, relu_116, mean_28, convolution_148, relu_117, div_28, sum_87, convolution_150, squeeze_364, relu_118, convolution_151, squeeze_367, relu_119, convolution_152, squeeze_370, relu_120, mean_29, convolution_153, relu_121, div_29, sum_90, convolution_155, squeeze_376, relu_122, convolution_156, squeeze_379, relu_123, convolution_157, squeeze_382, relu_124, mean_30, convolution_158, relu_125, div_30, sum_93, avg_pool2d_4, convolution_160, squeeze_388, avg_pool2d_5, convolution_161, squeeze_391, relu_126, convolution_162, squeeze_394, relu_127, convolution_163, squeeze_397, relu_128, mean_31, convolution_164, relu_129, div_31, sum_96, convolution_166, squeeze_403, relu_130, convolution_167, squeeze_406, relu_131, convolution_168, squeeze_409, relu_132, mean_32, convolution_169, relu_133, div_32, sum_99, convolution_171, squeeze_415, view_198, permute_34, le, unsqueeze_558, unsqueeze_584, unsqueeze_596, unsqueeze_608, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_696, unsqueeze_708, unsqueeze_720, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_796, unsqueeze_808, unsqueeze_820, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_896, unsqueeze_908, unsqueeze_920, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_996, unsqueeze_1008, unsqueeze_1020, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1096, unsqueeze_1108, unsqueeze_1120, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, unsqueeze_1196, unsqueeze_1208, unsqueeze_1220, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, unsqueeze_1296, unsqueeze_1308, unsqueeze_1320, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1396, unsqueeze_1408, unsqueeze_1420, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1496, unsqueeze_1508, unsqueeze_1520, unsqueeze_1546, unsqueeze_1558, unsqueeze_1570, unsqueeze_1596, unsqueeze_1608, unsqueeze_1620, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, unsqueeze_1696, unsqueeze_1708, unsqueeze_1720, unsqueeze_1746, unsqueeze_1758, unsqueeze_1770, unsqueeze_1796, unsqueeze_1808, unsqueeze_1820, unsqueeze_1832, unsqueeze_1858, unsqueeze_1870, unsqueeze_1882, unsqueeze_1908, unsqueeze_1920, unsqueeze_1932, unsqueeze_1958, unsqueeze_1970, unsqueeze_1982, unsqueeze_2008, unsqueeze_2020, unsqueeze_2032, unsqueeze_2044, unsqueeze_2070, unsqueeze_2082, unsqueeze_2094, unsqueeze_2120, unsqueeze_2132, unsqueeze_2144, unsqueeze_2170, unsqueeze_2182, unsqueeze_2194, unsqueeze_2206, unsqueeze_2232, unsqueeze_2244, unsqueeze_2256, unsqueeze_2268, unsqueeze_2280, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_10, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_16, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_20, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_28, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_31, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_34, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_38, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_40, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_43, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_46, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_53, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_55, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_58, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_61, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_64, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_68, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_70, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_73, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_76, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_79, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_82, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_84, (64, ), (1, ))
    assert_size_stride(primals_86, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_88, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_91, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_94, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_97, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_101, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_103, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_106, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_109, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_112, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_116, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_118, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_121, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_124, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_127, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_131, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_133, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_137, (1024, ), (1, ))
    assert_size_stride(primals_139, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_142, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_145, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_147, (128, ), (1, ))
    assert_size_stride(primals_149, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_151, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_152, (1024, ), (1, ))
    assert_size_stride(primals_154, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_155, (256, ), (1, ))
    assert_size_stride(primals_157, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_158, (512, ), (1, ))
    assert_size_stride(primals_160, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_164, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_166, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_169, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_172, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_175, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_179, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_181, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_182, (1024, ), (1, ))
    assert_size_stride(primals_184, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_185, (256, ), (1, ))
    assert_size_stride(primals_187, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_188, (512, ), (1, ))
    assert_size_stride(primals_190, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_192, (128, ), (1, ))
    assert_size_stride(primals_194, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_196, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_199, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_202, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_205, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_207, (128, ), (1, ))
    assert_size_stride(primals_209, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_211, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_212, (1024, ), (1, ))
    assert_size_stride(primals_214, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_217, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_218, (512, ), (1, ))
    assert_size_stride(primals_220, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_222, (128, ), (1, ))
    assert_size_stride(primals_224, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_226, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_229, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_232, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_233, (512, ), (1, ))
    assert_size_stride(primals_235, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_237, (128, ), (1, ))
    assert_size_stride(primals_239, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_241, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_244, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_245, (256, ), (1, ))
    assert_size_stride(primals_247, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_248, (512, ), (1, ))
    assert_size_stride(primals_250, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_254, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_256, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_257, (1024, ), (1, ))
    assert_size_stride(primals_259, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_262, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_265, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_267, (128, ), (1, ))
    assert_size_stride(primals_269, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_271, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_272, (1024, ), (1, ))
    assert_size_stride(primals_274, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_275, (256, ), (1, ))
    assert_size_stride(primals_277, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_280, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_282, (128, ), (1, ))
    assert_size_stride(primals_284, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_286, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_287, (1024, ), (1, ))
    assert_size_stride(primals_289, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_290, (256, ), (1, ))
    assert_size_stride(primals_292, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_295, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_299, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_301, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_304, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_305, (256, ), (1, ))
    assert_size_stride(primals_307, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_310, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_312, (128, ), (1, ))
    assert_size_stride(primals_314, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_316, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_317, (1024, ), (1, ))
    assert_size_stride(primals_319, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_320, (256, ), (1, ))
    assert_size_stride(primals_322, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_323, (512, ), (1, ))
    assert_size_stride(primals_325, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_327, (128, ), (1, ))
    assert_size_stride(primals_329, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_331, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_332, (1024, ), (1, ))
    assert_size_stride(primals_334, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_335, (256, ), (1, ))
    assert_size_stride(primals_337, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_338, (512, ), (1, ))
    assert_size_stride(primals_340, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_342, (128, ), (1, ))
    assert_size_stride(primals_344, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_346, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_347, (1024, ), (1, ))
    assert_size_stride(primals_349, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_350, (256, ), (1, ))
    assert_size_stride(primals_352, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_353, (512, ), (1, ))
    assert_size_stride(primals_355, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_357, (128, ), (1, ))
    assert_size_stride(primals_359, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_361, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_362, (1024, ), (1, ))
    assert_size_stride(primals_364, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_365, (256, ), (1, ))
    assert_size_stride(primals_367, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_368, (512, ), (1, ))
    assert_size_stride(primals_370, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_372, (128, ), (1, ))
    assert_size_stride(primals_374, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_376, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_377, (1024, ), (1, ))
    assert_size_stride(primals_379, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_380, (256, ), (1, ))
    assert_size_stride(primals_382, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_383, (512, ), (1, ))
    assert_size_stride(primals_385, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_387, (128, ), (1, ))
    assert_size_stride(primals_389, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_391, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_392, (1024, ), (1, ))
    assert_size_stride(primals_394, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_395, (256, ), (1, ))
    assert_size_stride(primals_397, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_398, (512, ), (1, ))
    assert_size_stride(primals_400, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_402, (128, ), (1, ))
    assert_size_stride(primals_404, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_406, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_407, (1024, ), (1, ))
    assert_size_stride(primals_409, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_410, (256, ), (1, ))
    assert_size_stride(primals_412, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_413, (512, ), (1, ))
    assert_size_stride(primals_415, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_417, (128, ), (1, ))
    assert_size_stride(primals_419, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_421, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_422, (1024, ), (1, ))
    assert_size_stride(primals_424, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_425, (256, ), (1, ))
    assert_size_stride(primals_427, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_428, (512, ), (1, ))
    assert_size_stride(primals_430, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_432, (128, ), (1, ))
    assert_size_stride(primals_434, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_436, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_437, (1024, ), (1, ))
    assert_size_stride(primals_439, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_440, (256, ), (1, ))
    assert_size_stride(primals_442, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_443, (512, ), (1, ))
    assert_size_stride(primals_445, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_447, (128, ), (1, ))
    assert_size_stride(primals_449, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_451, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_452, (1024, ), (1, ))
    assert_size_stride(primals_454, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_455, (256, ), (1, ))
    assert_size_stride(primals_457, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_458, (512, ), (1, ))
    assert_size_stride(primals_460, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_462, (128, ), (1, ))
    assert_size_stride(primals_464, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_466, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_467, (1024, ), (1, ))
    assert_size_stride(primals_469, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_470, (512, ), (1, ))
    assert_size_stride(primals_472, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_473, (1024, ), (1, ))
    assert_size_stride(primals_475, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_477, (256, ), (1, ))
    assert_size_stride(primals_479, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_481, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_482, (2048, ), (1, ))
    assert_size_stride(primals_484, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_485, (2048, ), (1, ))
    assert_size_stride(primals_487, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_488, (512, ), (1, ))
    assert_size_stride(primals_490, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_491, (1024, ), (1, ))
    assert_size_stride(primals_493, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_495, (256, ), (1, ))
    assert_size_stride(primals_497, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_499, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_500, (2048, ), (1, ))
    assert_size_stride(primals_502, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_503, (512, ), (1, ))
    assert_size_stride(primals_505, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_506, (1024, ), (1, ))
    assert_size_stride(primals_508, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_510, (256, ), (1, ))
    assert_size_stride(primals_512, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_514, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_515, (2048, ), (1, ))
    assert_size_stride(primals_936, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(convolution, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(convolution_1, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(convolution_2, (8, 128, 128, 128), (2097152, 16384, 128, 1))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(relu_2, (8, 128, 128, 128), (2097152, 16384, 128, 1))
    assert_size_stride(getitem_6, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(getitem_7, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(convolution_3, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(relu_3, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_4, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(squeeze_13, (128, ), (1, ))
    assert_size_stride(relu_4, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(mean, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(convolution_5, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(relu_5, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(div, (8, 2, 1, 64), (128, 64, 64, 1))
    assert_size_stride(sum_3, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_7, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(convolution_8, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(relu_6, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_7, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_10, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(squeeze_28, (128, ), (1, ))
    assert_size_stride(relu_8, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(mean_1, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(convolution_11, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(relu_9, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(div_1, (8, 2, 1, 64), (128, 64, 64, 1))
    assert_size_stride(sum_6, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_13, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(relu_10, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_14, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_37, (64, ), (1, ))
    assert_size_stride(relu_11, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_15, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(squeeze_40, (128, ), (1, ))
    assert_size_stride(relu_12, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(mean_2, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(convolution_16, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(relu_13, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(div_2, (8, 2, 1, 64), (128, 64, 64, 1))
    assert_size_stride(sum_9, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_18, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_46, (256, ), (1, ))
    assert_size_stride(relu_14, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_19, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(squeeze_49, (128, ), (1, ))
    assert_size_stride(relu_15, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(convolution_20, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_52, (256, ), (1, ))
    assert_size_stride(relu_16, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(mean_3, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_21, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(relu_17, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(div_3, (8, 2, 1, 128), (256, 128, 128, 1))
    assert_size_stride(sum_12, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(avg_pool2d, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_23, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_58, (512, ), (1, ))
    assert_size_stride(avg_pool2d_1, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(convolution_24, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_61, (512, ), (1, ))
    assert_size_stride(relu_18, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(convolution_25, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(relu_19, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_26, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(squeeze_67, (256, ), (1, ))
    assert_size_stride(relu_20, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(mean_4, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_27, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(relu_21, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(div_4, (8, 2, 1, 128), (256, 128, 128, 1))
    assert_size_stride(sum_15, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_29, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_73, (512, ), (1, ))
    assert_size_stride(relu_22, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(convolution_30, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(squeeze_76, (128, ), (1, ))
    assert_size_stride(relu_23, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_31, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(squeeze_79, (256, ), (1, ))
    assert_size_stride(relu_24, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(mean_5, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_32, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(relu_25, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(div_5, (8, 2, 1, 128), (256, 128, 128, 1))
    assert_size_stride(sum_18, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_34, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_85, (512, ), (1, ))
    assert_size_stride(relu_26, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(convolution_35, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(squeeze_88, (128, ), (1, ))
    assert_size_stride(relu_27, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_36, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(squeeze_91, (256, ), (1, ))
    assert_size_stride(relu_28, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(mean_6, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(convolution_37, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(relu_29, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(div_6, (8, 2, 1, 128), (256, 128, 128, 1))
    assert_size_stride(sum_21, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_39, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_97, (512, ), (1, ))
    assert_size_stride(relu_30, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(convolution_40, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(squeeze_100, (256, ), (1, ))
    assert_size_stride(relu_31, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(convolution_41, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_103, (512, ), (1, ))
    assert_size_stride(relu_32, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(mean_7, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_42, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_33, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_7, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_24, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(avg_pool2d_2, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_44, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_109, (1024, ), (1, ))
    assert_size_stride(avg_pool2d_3, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(convolution_45, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_112, (1024, ), (1, ))
    assert_size_stride(relu_34, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_46, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(relu_35, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_47, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_118, (512, ), (1, ))
    assert_size_stride(relu_36, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_8, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_48, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_37, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_8, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_27, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_50, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_124, (1024, ), (1, ))
    assert_size_stride(relu_38, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_51, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_127, (256, ), (1, ))
    assert_size_stride(relu_39, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_52, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_130, (512, ), (1, ))
    assert_size_stride(relu_40, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_9, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_53, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_41, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_9, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_30, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_55, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_136, (1024, ), (1, ))
    assert_size_stride(relu_42, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_56, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_139, (256, ), (1, ))
    assert_size_stride(relu_43, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_57, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_142, (512, ), (1, ))
    assert_size_stride(relu_44, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_10, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_58, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_45, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_10, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_33, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_60, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_148, (1024, ), (1, ))
    assert_size_stride(relu_46, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_61, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_151, (256, ), (1, ))
    assert_size_stride(relu_47, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_62, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_154, (512, ), (1, ))
    assert_size_stride(relu_48, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_11, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_63, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_49, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_11, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_36, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_65, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_160, (1024, ), (1, ))
    assert_size_stride(relu_50, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_66, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_163, (256, ), (1, ))
    assert_size_stride(relu_51, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_67, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_166, (512, ), (1, ))
    assert_size_stride(relu_52, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_12, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_68, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_53, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_12, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_39, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_70, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_172, (1024, ), (1, ))
    assert_size_stride(relu_54, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_71, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_175, (256, ), (1, ))
    assert_size_stride(relu_55, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_72, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_178, (512, ), (1, ))
    assert_size_stride(relu_56, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_13, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_73, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_57, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_13, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_42, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_75, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_184, (1024, ), (1, ))
    assert_size_stride(relu_58, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_76, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_187, (256, ), (1, ))
    assert_size_stride(relu_59, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_77, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_190, (512, ), (1, ))
    assert_size_stride(relu_60, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_14, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_78, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_61, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_14, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_45, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_80, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_196, (1024, ), (1, ))
    assert_size_stride(relu_62, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_81, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_199, (256, ), (1, ))
    assert_size_stride(relu_63, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_82, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_202, (512, ), (1, ))
    assert_size_stride(relu_64, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_15, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_83, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_65, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_15, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_48, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_85, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_208, (1024, ), (1, ))
    assert_size_stride(relu_66, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_86, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_211, (256, ), (1, ))
    assert_size_stride(relu_67, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_87, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_214, (512, ), (1, ))
    assert_size_stride(relu_68, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_16, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_88, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_69, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_16, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_51, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_90, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_220, (1024, ), (1, ))
    assert_size_stride(relu_70, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_91, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_223, (256, ), (1, ))
    assert_size_stride(relu_71, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_92, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_226, (512, ), (1, ))
    assert_size_stride(relu_72, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_17, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_93, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_73, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_17, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_54, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_95, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_232, (1024, ), (1, ))
    assert_size_stride(relu_74, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_96, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_235, (256, ), (1, ))
    assert_size_stride(relu_75, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_97, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_238, (512, ), (1, ))
    assert_size_stride(relu_76, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_18, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_98, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_77, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_18, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_57, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_100, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_244, (1024, ), (1, ))
    assert_size_stride(relu_78, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_101, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_247, (256, ), (1, ))
    assert_size_stride(relu_79, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_102, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_250, (512, ), (1, ))
    assert_size_stride(relu_80, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_19, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_103, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_81, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_19, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_60, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_105, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_256, (1024, ), (1, ))
    assert_size_stride(relu_82, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_106, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_259, (256, ), (1, ))
    assert_size_stride(relu_83, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_107, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_262, (512, ), (1, ))
    assert_size_stride(relu_84, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_20, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_108, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_85, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_20, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_63, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_110, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_268, (1024, ), (1, ))
    assert_size_stride(relu_86, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_111, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_271, (256, ), (1, ))
    assert_size_stride(relu_87, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_112, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_274, (512, ), (1, ))
    assert_size_stride(relu_88, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_21, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_113, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_89, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_21, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_66, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_115, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_280, (1024, ), (1, ))
    assert_size_stride(relu_90, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_116, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_283, (256, ), (1, ))
    assert_size_stride(relu_91, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_117, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_286, (512, ), (1, ))
    assert_size_stride(relu_92, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_22, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_118, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_93, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_22, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_69, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_120, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_292, (1024, ), (1, ))
    assert_size_stride(relu_94, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_121, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_295, (256, ), (1, ))
    assert_size_stride(relu_95, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_122, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_298, (512, ), (1, ))
    assert_size_stride(relu_96, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_23, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_123, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_97, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_23, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_72, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_125, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_304, (1024, ), (1, ))
    assert_size_stride(relu_98, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_126, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_307, (256, ), (1, ))
    assert_size_stride(relu_99, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_127, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_310, (512, ), (1, ))
    assert_size_stride(relu_100, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_24, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_128, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_101, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_24, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_75, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_130, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_316, (1024, ), (1, ))
    assert_size_stride(relu_102, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_131, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_319, (256, ), (1, ))
    assert_size_stride(relu_103, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_132, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_322, (512, ), (1, ))
    assert_size_stride(relu_104, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_25, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_133, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_105, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_25, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_78, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_135, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_328, (1024, ), (1, ))
    assert_size_stride(relu_106, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_136, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_331, (256, ), (1, ))
    assert_size_stride(relu_107, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_137, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_334, (512, ), (1, ))
    assert_size_stride(relu_108, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_26, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_138, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_109, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_26, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_81, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_140, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_340, (1024, ), (1, ))
    assert_size_stride(relu_110, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_141, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_343, (256, ), (1, ))
    assert_size_stride(relu_111, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_142, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_346, (512, ), (1, ))
    assert_size_stride(relu_112, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_27, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_143, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_113, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_27, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_84, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_145, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_352, (1024, ), (1, ))
    assert_size_stride(relu_114, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_146, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_355, (256, ), (1, ))
    assert_size_stride(relu_115, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_147, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_358, (512, ), (1, ))
    assert_size_stride(relu_116, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_28, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_148, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_117, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_28, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_87, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_150, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_364, (1024, ), (1, ))
    assert_size_stride(relu_118, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_151, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_367, (256, ), (1, ))
    assert_size_stride(relu_119, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_152, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_370, (512, ), (1, ))
    assert_size_stride(relu_120, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(mean_29, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(convolution_153, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(relu_121, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_29, (8, 2, 1, 256), (512, 256, 256, 1))
    assert_size_stride(sum_90, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_155, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_376, (1024, ), (1, ))
    assert_size_stride(relu_122, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_156, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_379, (512, ), (1, ))
    assert_size_stride(relu_123, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(convolution_157, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_382, (1024, ), (1, ))
    assert_size_stride(relu_124, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(mean_30, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_158, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(relu_125, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(div_30, (8, 2, 1, 512), (1024, 512, 512, 1))
    assert_size_stride(sum_93, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(avg_pool2d_4, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(convolution_160, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(squeeze_388, (2048, ), (1, ))
    assert_size_stride(avg_pool2d_5, (8, 1024, 8, 8), (65536, 64, 8, 1))
    assert_size_stride(convolution_161, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(squeeze_391, (2048, ), (1, ))
    assert_size_stride(relu_126, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(convolution_162, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(squeeze_394, (512, ), (1, ))
    assert_size_stride(relu_127, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(convolution_163, (8, 1024, 8, 8), (65536, 64, 8, 1))
    assert_size_stride(squeeze_397, (1024, ), (1, ))
    assert_size_stride(relu_128, (8, 1024, 8, 8), (65536, 64, 8, 1))
    assert_size_stride(mean_31, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_164, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(relu_129, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(div_31, (8, 2, 1, 512), (1024, 512, 512, 1))
    assert_size_stride(sum_96, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(convolution_166, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(squeeze_403, (2048, ), (1, ))
    assert_size_stride(relu_130, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(convolution_167, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(squeeze_406, (512, ), (1, ))
    assert_size_stride(relu_131, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(convolution_168, (8, 1024, 8, 8), (65536, 64, 8, 1))
    assert_size_stride(squeeze_409, (1024, ), (1, ))
    assert_size_stride(relu_132, (8, 1024, 8, 8), (65536, 64, 8, 1))
    assert_size_stride(mean_32, (8, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(convolution_169, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(relu_133, (8, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(div_32, (8, 2, 1, 512), (1024, 512, 512, 1))
    assert_size_stride(sum_99, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(convolution_171, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(squeeze_415, (2048, ), (1, ))
    assert_size_stride(view_198, (8, 2048), (2048, 1))
    assert_size_stride(permute_34, (1000, 2048), (2048, 1))
    assert_size_stride(le, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(unsqueeze_558, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_584, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_596, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_608, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_696, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_708, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_720, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_770, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_796, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_808, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_820, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_870, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_896, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_908, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_920, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_946, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_958, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_970, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_996, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1008, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1020, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1046, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1058, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1070, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1096, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1108, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1120, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1146, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1158, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1170, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1196, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1208, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1220, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1246, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1258, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1270, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1296, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1308, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1320, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1346, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1358, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1370, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1396, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1408, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1420, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1446, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1458, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1470, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1496, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1508, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1520, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1546, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1558, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1570, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1596, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1608, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1620, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1646, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1658, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1670, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1696, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1708, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1720, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1746, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1758, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1770, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1796, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1808, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1820, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1832, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1858, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1870, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1882, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1908, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1920, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1932, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1958, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1970, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1982, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_2008, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_2020, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_2032, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_2044, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_2070, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_2082, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_2094, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_2120, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_2132, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_2144, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_2170, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_2182, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_2194, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_2206, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_2232, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_2244, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_2256, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_2268, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_2280, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf99 = empty((8, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_34, out=buf99)
        del permute_34
        buf102 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf103 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf104 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_0.run(le, buf99, convolution_171, unsqueeze_558, squeeze_415, buf102, buf103, buf104, 2048, 512, grid=grid(2048), stream=stream0)
        buf105 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_1.run(le, buf99, convolution_171, unsqueeze_558, buf103, squeeze_415, buf102, primals_515, buf105, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_171
        del primals_515
        del squeeze_415
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf106 = aten.convolution_backward(buf105, sum_99, primals_514, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_514
        del sum_99
        buf107 = buf106[0]
        buf109 = empty((8, 2, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_2.run(buf107, relu_132, buf109, 8192, 64, grid=grid(8192), stream=stream0)
        buf110 = empty((8, 2, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_3.run(buf109, div_32, buf110, 8192, grid=grid(8192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf112 = aten.convolution_backward(reinterpret_tensor(buf110, (8, 1024, 1, 1), (1024, 1, 0, 0), 0), relu_133, primals_512, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_512
        buf113 = buf112[0]
        buf96 = empty((1, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf115 = empty((256, ), device='cuda', dtype=torch.float32)
        buf116 = empty((256, ), device='cuda', dtype=torch.float32)
        buf97 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf118 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_163], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4.run(convolution_169, relu_133, buf113, buf96, buf115, buf116, buf97, buf118, 256, 8, grid=grid(256), stream=stream0)
        buf117 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_5.run(buf117, relu_133, convolution_169, buf96, buf116, buf97, buf115, primals_510, 2048, grid=grid(2048), stream=stream0)
        del convolution_169
        del primals_510
        del relu_133
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf120 = aten.convolution_backward(buf117, mean_32, primals_508, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_32
        del primals_508
        buf121 = buf120[0]
        buf123 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf124 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf126 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(relu_132, buf107, div_32, buf121, convolution_168, unsqueeze_584, squeeze_409, buf123, buf124, buf126, 1024, 512, grid=grid(1024), stream=stream0)
        buf125 = empty((8, 1024, 8, 8), device='cuda', dtype=torch.float32)
        buf127 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7.run(buf127, relu_132, buf107, div_32, buf121, convolution_168, unsqueeze_584, buf124, squeeze_409, buf123, primals_506, 524288, grid=grid(524288), stream=stream0)
        del buf107
        del convolution_168
        del div_32
        del primals_506
        del relu_132
        del squeeze_409
        del unsqueeze_584
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf128 = aten.convolution_backward(buf127, relu_131, primals_505, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_505
        buf129 = buf128[0]
        buf131 = empty((512, ), device='cuda', dtype=torch.float32)
        buf132 = empty((512, ), device='cuda', dtype=torch.float32)
        buf133 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(relu_131, buf129, convolution_167, unsqueeze_596, squeeze_406, buf131, buf132, buf133, 512, 512, grid=grid(512), stream=stream0)
        buf134 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9.run(buf134, relu_131, convolution_167, unsqueeze_596, buf132, squeeze_406, buf131, primals_503, 262144, grid=grid(262144), stream=stream0)
        del convolution_167
        del primals_503
        del relu_131
        del squeeze_406
        del unsqueeze_596
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf135 = aten.convolution_backward(buf134, relu_130, primals_502, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf134
        del primals_502
        buf136 = buf135[0]
        buf138 = buf103; del buf103  # reuse
        buf139 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf141 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_10.run(relu_130, le, buf99, buf136, convolution_166, unsqueeze_608, squeeze_403, buf138, buf139, buf141, 2048, 512, grid=grid(2048), stream=stream0)
        buf140 = buf105; del buf105  # reuse
        buf142 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_11.run(buf142, relu_130, le, buf99, buf136, convolution_166, unsqueeze_608, buf139, squeeze_403, buf138, primals_500, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_166
        del primals_500
        del squeeze_403
        del unsqueeze_608
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf143 = aten.convolution_backward(buf142, sum_96, primals_499, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_499
        del sum_96
        buf144 = buf143[0]
        buf146 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_2.run(buf144, relu_128, buf146, 8192, 64, grid=grid(8192), stream=stream0)
        buf147 = empty((8, 2, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_3.run(buf146, div_31, buf147, 8192, grid=grid(8192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf149 = aten.convolution_backward(reinterpret_tensor(buf147, (8, 1024, 1, 1), (1024, 1, 0, 0), 0), relu_129, primals_497, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_497
        buf150 = buf149[0]
        buf93 = reinterpret_tensor(buf97, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf97  # reuse
        buf152 = reinterpret_tensor(buf96, (256, ), (1, ), 0); del buf96  # reuse
        buf153 = buf116; del buf116  # reuse
        buf94 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf155 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_158], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4.run(convolution_164, relu_129, buf150, buf93, buf152, buf153, buf94, buf155, 256, 8, grid=grid(256), stream=stream0)
        buf154 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_5.run(buf154, relu_129, convolution_164, buf93, buf153, buf94, buf152, primals_495, 2048, grid=grid(2048), stream=stream0)
        del convolution_164
        del primals_495
        del relu_129
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf157 = aten.convolution_backward(buf154, mean_31, primals_493, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_31
        del primals_493
        buf158 = buf157[0]
        buf160 = buf124; del buf124  # reuse
        buf161 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf163 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(relu_128, buf144, div_31, buf158, convolution_163, unsqueeze_634, squeeze_397, buf160, buf161, buf163, 1024, 512, grid=grid(1024), stream=stream0)
        buf162 = buf127; del buf127  # reuse
        buf164 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7.run(buf164, relu_128, buf144, div_31, buf158, convolution_163, unsqueeze_634, buf161, squeeze_397, buf160, primals_491, 524288, grid=grid(524288), stream=stream0)
        del buf144
        del convolution_163
        del div_31
        del primals_491
        del relu_128
        del squeeze_397
        del unsqueeze_634
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf165 = aten.convolution_backward(buf164, relu_127, primals_490, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf164
        del primals_490
        buf166 = buf165[0]
        buf168 = buf132; del buf132  # reuse
        buf169 = empty((512, ), device='cuda', dtype=torch.float32)
        buf170 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(relu_127, buf166, convolution_162, unsqueeze_646, squeeze_394, buf168, buf169, buf170, 512, 512, grid=grid(512), stream=stream0)
        buf171 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9.run(buf171, relu_127, convolution_162, unsqueeze_646, buf169, squeeze_394, buf168, primals_488, 262144, grid=grid(262144), stream=stream0)
        del convolution_162
        del primals_488
        del relu_127
        del squeeze_394
        del unsqueeze_646
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf172 = aten.convolution_backward(buf171, relu_126, primals_487, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf171
        del primals_487
        buf173 = buf172[0]
        buf175 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
        triton_poi_fused_add_div_threshold_backward_12.run(buf175, relu_126, relu_130, le, buf99, buf173, 1048576, grid=grid(1048576), stream=stream0)
        del buf99
        del le
        del relu_126
        del relu_130
        buf176 = buf139; del buf139  # reuse
        buf177 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf183 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf178 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf184 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_13.run(buf175, convolution_161, unsqueeze_658, convolution_160, unsqueeze_670, squeeze_391, squeeze_388, buf176, buf177, buf183, buf178, buf184, 2048, 512, grid=grid(2048), stream=stream0)
        buf179 = buf173; del buf173  # reuse
        buf185 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_14.run(buf175, convolution_161, unsqueeze_658, buf177, squeeze_391, buf176, primals_485, convolution_160, unsqueeze_670, buf183, squeeze_388, primals_482, buf179, buf185, 1048576, grid=grid(1048576), stream=stream0)
        del buf175
        del buf177
        del buf183
        del convolution_160
        del convolution_161
        del primals_482
        del primals_485
        del squeeze_388
        del squeeze_391
        del unsqueeze_658
        del unsqueeze_670
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf180 = aten.convolution_backward(buf179, avg_pool2d_5, primals_484, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_5
        del buf179
        del primals_484
        buf181 = buf180[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf186 = aten.convolution_backward(buf185, avg_pool2d_4, primals_481, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_4
        del primals_481
        buf187 = buf186[0]
        buf189 = reinterpret_tensor(buf185, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_15.run(buf187, buf189, 1048576, grid=grid(1048576), stream=stream0)
        del buf187
        buf190 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_16.run(buf189, relu_124, buf190, 8192, 256, grid=grid(8192), stream=stream0)
        buf191 = empty((8, 2, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_3.run(buf190, div_30, buf191, 8192, grid=grid(8192), stream=stream0)
        del buf190
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf193 = aten.convolution_backward(reinterpret_tensor(buf191, (8, 1024, 1, 1), (1024, 1, 0, 0), 0), relu_125, primals_479, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_479
        buf194 = buf193[0]
        buf91 = buf94; del buf94  # reuse
        buf90 = buf93; del buf93  # reuse
        buf196 = buf153; del buf153  # reuse
        buf197 = empty((256, ), device='cuda', dtype=torch.float32)
        buf199 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_153], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_17.run(convolution_158, relu_125, buf194, buf91, buf90, buf196, buf197, buf199, 256, 8, grid=grid(256), stream=stream0)
        buf198 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_5.run(buf198, relu_125, convolution_158, buf90, buf197, buf91, buf196, primals_477, 2048, grid=grid(2048), stream=stream0)
        del convolution_158
        del primals_477
        del relu_125
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf201 = aten.convolution_backward(buf198, mean_30, primals_475, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_30
        del primals_475
        buf202 = buf201[0]
        buf204 = buf161; del buf161  # reuse
        buf205 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf207 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_124, buf189, div_30, buf202, convolution_157, unsqueeze_696, squeeze_382, buf204, buf205, buf207, 1024, 2048, grid=grid(1024), stream=stream0)
        buf206 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        buf208 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(buf208, relu_124, buf189, div_30, buf202, convolution_157, unsqueeze_696, buf205, squeeze_382, buf204, primals_473, 2097152, grid=grid(2097152), stream=stream0)
        del buf189
        del convolution_157
        del div_30
        del primals_473
        del relu_124
        del squeeze_382
        del unsqueeze_696
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf209 = aten.convolution_backward(buf208, relu_123, primals_472, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_472
        buf210 = buf209[0]
        buf212 = buf169; del buf169  # reuse
        buf213 = empty((512, ), device='cuda', dtype=torch.float32)
        buf214 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_20.run(relu_123, buf210, convolution_156, unsqueeze_708, squeeze_379, buf212, buf213, buf214, 512, 2048, grid=grid(512), stream=stream0)
        buf215 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(buf215, relu_123, convolution_156, unsqueeze_708, buf213, squeeze_379, buf212, primals_470, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_156
        del primals_470
        del relu_123
        del squeeze_379
        del unsqueeze_708
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf216 = aten.convolution_backward(buf215, relu_122, primals_469, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_469
        buf217 = buf216[0]
        buf219 = buf205; del buf205  # reuse
        buf220 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf222 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_22.run(relu_122, buf181, buf217, convolution_155, unsqueeze_720, squeeze_376, buf219, buf220, buf222, 1024, 2048, grid=grid(1024), stream=stream0)
        buf221 = buf208; del buf208  # reuse
        buf223 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_23.run(buf223, relu_122, buf181, buf217, convolution_155, unsqueeze_720, buf220, squeeze_376, buf219, primals_467, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_155
        del primals_467
        del squeeze_376
        del unsqueeze_720
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf224 = aten.convolution_backward(buf223, sum_90, primals_466, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf223
        del primals_466
        del sum_90
        buf225 = buf224[0]
        buf227 = reinterpret_tensor(buf202, (8, 2, 256, 1, 1), (512, 256, 1, 1, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf225, relu_120, buf227, 4096, 256, grid=grid(4096), stream=stream0)
        buf228 = reinterpret_tensor(buf158, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf227, div_29, buf228, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf230 = aten.convolution_backward(reinterpret_tensor(buf228, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_121, primals_464, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_464
        buf231 = buf230[0]
        buf88 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf87 = empty((1, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf233 = empty((128, ), device='cuda', dtype=torch.float32)
        buf234 = empty((128, ), device='cuda', dtype=torch.float32)
        buf236 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_153, relu_121, buf231, buf88, buf87, buf233, buf234, buf236, 128, 8, grid=grid(128), stream=stream0)
        buf235 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf235, relu_121, convolution_153, buf87, buf234, buf88, buf233, primals_462, 1024, grid=grid(1024), stream=stream0)
        del convolution_153
        del primals_462
        del relu_121
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf238 = aten.convolution_backward(buf235, mean_29, primals_460, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_29
        del primals_460
        buf239 = buf238[0]
        buf241 = buf213; del buf213  # reuse
        buf242 = empty((512, ), device='cuda', dtype=torch.float32)
        buf244 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_120, buf225, div_29, buf239, convolution_152, unsqueeze_746, squeeze_370, buf241, buf242, buf244, 512, 2048, grid=grid(512), stream=stream0)
        buf243 = buf215; del buf215  # reuse
        buf245 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf245, relu_120, buf225, div_29, buf239, convolution_152, unsqueeze_746, buf242, squeeze_370, buf241, primals_458, 1048576, grid=grid(1048576), stream=stream0)
        del buf225
        del buf239
        del convolution_152
        del div_29
        del primals_458
        del relu_120
        del squeeze_370
        del unsqueeze_746
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf246 = aten.convolution_backward(buf245, relu_119, primals_457, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_457
        buf247 = buf246[0]
        buf249 = reinterpret_tensor(buf91, (256, ), (1, ), 0); del buf91  # reuse
        buf250 = reinterpret_tensor(buf90, (256, ), (1, ), 0); del buf90  # reuse
        buf251 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_119, buf247, convolution_151, unsqueeze_758, squeeze_367, buf249, buf250, buf251, 256, 2048, grid=grid(256), stream=stream0)
        buf252 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf252, relu_119, convolution_151, unsqueeze_758, buf250, squeeze_367, buf249, primals_455, 524288, grid=grid(524288), stream=stream0)
        del convolution_151
        del primals_455
        del relu_119
        del squeeze_367
        del unsqueeze_758
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf253 = aten.convolution_backward(buf252, relu_118, primals_454, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf252
        del primals_454
        buf254 = buf253[0]
        buf256 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_threshold_backward_32.run(buf256, relu_118, relu_122, buf181, buf254, 2097152, grid=grid(2097152), stream=stream0)
        del buf181
        del relu_118
        del relu_122
        buf257 = buf220; del buf220  # reuse
        buf258 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf259 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf256, convolution_150, unsqueeze_770, squeeze_364, buf257, buf258, buf259, 1024, 2048, grid=grid(1024), stream=stream0)
        buf260 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf256, convolution_150, unsqueeze_770, buf258, squeeze_364, buf257, primals_452, buf260, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_150
        del primals_452
        del squeeze_364
        del unsqueeze_770
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf261 = aten.convolution_backward(buf260, sum_87, primals_451, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_451
        del sum_87
        buf262 = buf261[0]
        buf264 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf262, relu_116, buf264, 4096, 256, grid=grid(4096), stream=stream0)
        buf265 = reinterpret_tensor(buf121, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf264, div_28, buf265, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf267 = aten.convolution_backward(reinterpret_tensor(buf265, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_117, primals_449, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_449
        buf268 = buf267[0]
        buf85 = buf88; del buf88  # reuse
        buf84 = buf87; del buf87  # reuse
        buf270 = buf234; del buf234  # reuse
        buf271 = empty((128, ), device='cuda', dtype=torch.float32)
        buf273 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_143], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_148, relu_117, buf268, buf85, buf84, buf270, buf271, buf273, 128, 8, grid=grid(128), stream=stream0)
        buf272 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf272, relu_117, convolution_148, buf84, buf271, buf85, buf270, primals_447, 1024, grid=grid(1024), stream=stream0)
        del convolution_148
        del primals_447
        del relu_117
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf275 = aten.convolution_backward(buf272, mean_28, primals_445, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_28
        del primals_445
        buf276 = buf275[0]
        buf278 = buf242; del buf242  # reuse
        buf279 = empty((512, ), device='cuda', dtype=torch.float32)
        buf281 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_116, buf262, div_28, buf276, convolution_147, unsqueeze_796, squeeze_358, buf278, buf279, buf281, 512, 2048, grid=grid(512), stream=stream0)
        buf280 = buf245; del buf245  # reuse
        buf282 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf282, relu_116, buf262, div_28, buf276, convolution_147, unsqueeze_796, buf279, squeeze_358, buf278, primals_443, 1048576, grid=grid(1048576), stream=stream0)
        del buf262
        del buf276
        del convolution_147
        del div_28
        del primals_443
        del relu_116
        del squeeze_358
        del unsqueeze_796
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf283 = aten.convolution_backward(buf282, relu_115, primals_442, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_442
        buf284 = buf283[0]
        buf286 = buf250; del buf250  # reuse
        buf287 = empty((256, ), device='cuda', dtype=torch.float32)
        buf288 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_115, buf284, convolution_146, unsqueeze_808, squeeze_355, buf286, buf287, buf288, 256, 2048, grid=grid(256), stream=stream0)
        buf289 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf289, relu_115, convolution_146, unsqueeze_808, buf287, squeeze_355, buf286, primals_440, 524288, grid=grid(524288), stream=stream0)
        del convolution_146
        del primals_440
        del relu_115
        del squeeze_355
        del unsqueeze_808
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf290 = aten.convolution_backward(buf289, relu_114, primals_439, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf289
        del primals_439
        buf291 = buf290[0]
        buf293 = buf258; del buf258  # reuse
        buf294 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf296 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_114, buf256, buf291, convolution_145, unsqueeze_820, squeeze_352, buf293, buf294, buf296, 1024, 2048, grid=grid(1024), stream=stream0)
        buf295 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_114, buf256, buf291, convolution_145, unsqueeze_820, buf294, squeeze_352, buf293, primals_437, buf295, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_145
        del primals_437
        del squeeze_352
        del unsqueeze_820
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf297 = aten.convolution_backward(buf295, sum_84, primals_436, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf295
        del primals_436
        del sum_84
        buf298 = buf297[0]
        buf300 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf298, relu_112, buf300, 4096, 256, grid=grid(4096), stream=stream0)
        buf301 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf300, div_27, buf301, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf303 = aten.convolution_backward(reinterpret_tensor(buf301, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_113, primals_434, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_434
        buf304 = buf303[0]
        buf82 = buf85; del buf85  # reuse
        buf81 = buf84; del buf84  # reuse
        buf306 = buf271; del buf271  # reuse
        buf307 = empty((128, ), device='cuda', dtype=torch.float32)
        buf309 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_143, relu_113, buf304, buf82, buf81, buf306, buf307, buf309, 128, 8, grid=grid(128), stream=stream0)
        buf308 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf308, relu_113, convolution_143, buf81, buf307, buf82, buf306, primals_432, 1024, grid=grid(1024), stream=stream0)
        del convolution_143
        del primals_432
        del relu_113
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf311 = aten.convolution_backward(buf308, mean_27, primals_430, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_27
        del primals_430
        buf312 = buf311[0]
        buf314 = buf279; del buf279  # reuse
        buf315 = empty((512, ), device='cuda', dtype=torch.float32)
        buf317 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_112, buf298, div_27, buf312, convolution_142, unsqueeze_846, squeeze_346, buf314, buf315, buf317, 512, 2048, grid=grid(512), stream=stream0)
        buf316 = buf282; del buf282  # reuse
        buf318 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf318, relu_112, buf298, div_27, buf312, convolution_142, unsqueeze_846, buf315, squeeze_346, buf314, primals_428, 1048576, grid=grid(1048576), stream=stream0)
        del buf298
        del buf312
        del convolution_142
        del div_27
        del primals_428
        del relu_112
        del squeeze_346
        del unsqueeze_846
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf319 = aten.convolution_backward(buf318, relu_111, primals_427, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_427
        buf320 = buf319[0]
        buf322 = buf287; del buf287  # reuse
        buf323 = empty((256, ), device='cuda', dtype=torch.float32)
        buf324 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_111, buf320, convolution_141, unsqueeze_858, squeeze_343, buf322, buf323, buf324, 256, 2048, grid=grid(256), stream=stream0)
        buf325 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf325, relu_111, convolution_141, unsqueeze_858, buf323, squeeze_343, buf322, primals_425, 524288, grid=grid(524288), stream=stream0)
        del convolution_141
        del primals_425
        del relu_111
        del squeeze_343
        del unsqueeze_858
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf326 = aten.convolution_backward(buf325, relu_110, primals_424, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf325
        del primals_424
        buf327 = buf326[0]
        buf329 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf329, relu_110, relu_114, buf291, buf327, 2097152, grid=grid(2097152), stream=stream0)
        del buf291
        del relu_110
        del relu_114
        buf330 = buf294; del buf294  # reuse
        buf331 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf332 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf329, convolution_140, unsqueeze_870, squeeze_340, buf330, buf331, buf332, 1024, 2048, grid=grid(1024), stream=stream0)
        buf333 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf329, convolution_140, unsqueeze_870, buf331, squeeze_340, buf330, primals_422, buf333, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_140
        del primals_422
        del squeeze_340
        del unsqueeze_870
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf334 = aten.convolution_backward(buf333, sum_81, primals_421, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_421
        del sum_81
        buf335 = buf334[0]
        buf337 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf335, relu_108, buf337, 4096, 256, grid=grid(4096), stream=stream0)
        buf338 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf337, div_26, buf338, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf340 = aten.convolution_backward(reinterpret_tensor(buf338, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_109, primals_419, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_419
        buf341 = buf340[0]
        buf79 = buf82; del buf82  # reuse
        buf78 = buf81; del buf81  # reuse
        buf343 = buf307; del buf307  # reuse
        buf344 = empty((128, ), device='cuda', dtype=torch.float32)
        buf346 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_138, relu_109, buf341, buf79, buf78, buf343, buf344, buf346, 128, 8, grid=grid(128), stream=stream0)
        buf345 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf345, relu_109, convolution_138, buf78, buf344, buf79, buf343, primals_417, 1024, grid=grid(1024), stream=stream0)
        del convolution_138
        del primals_417
        del relu_109
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf348 = aten.convolution_backward(buf345, mean_26, primals_415, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_26
        del primals_415
        buf349 = buf348[0]
        buf351 = buf315; del buf315  # reuse
        buf352 = empty((512, ), device='cuda', dtype=torch.float32)
        buf354 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_108, buf335, div_26, buf349, convolution_137, unsqueeze_896, squeeze_334, buf351, buf352, buf354, 512, 2048, grid=grid(512), stream=stream0)
        buf353 = buf318; del buf318  # reuse
        buf355 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf355, relu_108, buf335, div_26, buf349, convolution_137, unsqueeze_896, buf352, squeeze_334, buf351, primals_413, 1048576, grid=grid(1048576), stream=stream0)
        del buf335
        del buf349
        del convolution_137
        del div_26
        del primals_413
        del relu_108
        del squeeze_334
        del unsqueeze_896
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf356 = aten.convolution_backward(buf355, relu_107, primals_412, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_412
        buf357 = buf356[0]
        buf359 = buf323; del buf323  # reuse
        buf360 = empty((256, ), device='cuda', dtype=torch.float32)
        buf361 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_107, buf357, convolution_136, unsqueeze_908, squeeze_331, buf359, buf360, buf361, 256, 2048, grid=grid(256), stream=stream0)
        buf362 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf362, relu_107, convolution_136, unsqueeze_908, buf360, squeeze_331, buf359, primals_410, 524288, grid=grid(524288), stream=stream0)
        del convolution_136
        del primals_410
        del relu_107
        del squeeze_331
        del unsqueeze_908
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf363 = aten.convolution_backward(buf362, relu_106, primals_409, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf362
        del primals_409
        buf364 = buf363[0]
        buf366 = buf331; del buf331  # reuse
        buf367 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf369 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_106, buf329, buf364, convolution_135, unsqueeze_920, squeeze_328, buf366, buf367, buf369, 1024, 2048, grid=grid(1024), stream=stream0)
        buf368 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_106, buf329, buf364, convolution_135, unsqueeze_920, buf367, squeeze_328, buf366, primals_407, buf368, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_135
        del primals_407
        del squeeze_328
        del unsqueeze_920
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf370 = aten.convolution_backward(buf368, sum_78, primals_406, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf368
        del primals_406
        del sum_78
        buf371 = buf370[0]
        buf373 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf371, relu_104, buf373, 4096, 256, grid=grid(4096), stream=stream0)
        buf374 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf373, div_25, buf374, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf376 = aten.convolution_backward(reinterpret_tensor(buf374, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_105, primals_404, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_404
        buf377 = buf376[0]
        buf76 = buf79; del buf79  # reuse
        buf75 = buf78; del buf78  # reuse
        buf379 = buf344; del buf344  # reuse
        buf380 = empty((128, ), device='cuda', dtype=torch.float32)
        buf382 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_133, relu_105, buf377, buf76, buf75, buf379, buf380, buf382, 128, 8, grid=grid(128), stream=stream0)
        buf381 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf381, relu_105, convolution_133, buf75, buf380, buf76, buf379, primals_402, 1024, grid=grid(1024), stream=stream0)
        del convolution_133
        del primals_402
        del relu_105
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf384 = aten.convolution_backward(buf381, mean_25, primals_400, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_25
        del primals_400
        buf385 = buf384[0]
        buf387 = buf352; del buf352  # reuse
        buf388 = empty((512, ), device='cuda', dtype=torch.float32)
        buf390 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_104, buf371, div_25, buf385, convolution_132, unsqueeze_946, squeeze_322, buf387, buf388, buf390, 512, 2048, grid=grid(512), stream=stream0)
        buf389 = buf355; del buf355  # reuse
        buf391 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf391, relu_104, buf371, div_25, buf385, convolution_132, unsqueeze_946, buf388, squeeze_322, buf387, primals_398, 1048576, grid=grid(1048576), stream=stream0)
        del buf371
        del buf385
        del convolution_132
        del div_25
        del primals_398
        del relu_104
        del squeeze_322
        del unsqueeze_946
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf392 = aten.convolution_backward(buf391, relu_103, primals_397, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_397
        buf393 = buf392[0]
        buf395 = buf360; del buf360  # reuse
        buf396 = empty((256, ), device='cuda', dtype=torch.float32)
        buf397 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_103, buf393, convolution_131, unsqueeze_958, squeeze_319, buf395, buf396, buf397, 256, 2048, grid=grid(256), stream=stream0)
        buf398 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf398, relu_103, convolution_131, unsqueeze_958, buf396, squeeze_319, buf395, primals_395, 524288, grid=grid(524288), stream=stream0)
        del convolution_131
        del primals_395
        del relu_103
        del squeeze_319
        del unsqueeze_958
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf399 = aten.convolution_backward(buf398, relu_102, primals_394, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf398
        del primals_394
        buf400 = buf399[0]
        buf402 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf402, relu_102, relu_106, buf364, buf400, 2097152, grid=grid(2097152), stream=stream0)
        del buf364
        del relu_102
        del relu_106
        buf403 = buf367; del buf367  # reuse
        buf404 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf405 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf402, convolution_130, unsqueeze_970, squeeze_316, buf403, buf404, buf405, 1024, 2048, grid=grid(1024), stream=stream0)
        buf406 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf402, convolution_130, unsqueeze_970, buf404, squeeze_316, buf403, primals_392, buf406, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_130
        del primals_392
        del squeeze_316
        del unsqueeze_970
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf407 = aten.convolution_backward(buf406, sum_75, primals_391, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_391
        del sum_75
        buf408 = buf407[0]
        buf410 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf408, relu_100, buf410, 4096, 256, grid=grid(4096), stream=stream0)
        buf411 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf410, div_24, buf411, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf413 = aten.convolution_backward(reinterpret_tensor(buf411, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_101, primals_389, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_389
        buf414 = buf413[0]
        buf73 = buf76; del buf76  # reuse
        buf72 = buf75; del buf75  # reuse
        buf416 = buf380; del buf380  # reuse
        buf417 = empty((128, ), device='cuda', dtype=torch.float32)
        buf419 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_128, relu_101, buf414, buf73, buf72, buf416, buf417, buf419, 128, 8, grid=grid(128), stream=stream0)
        buf418 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf418, relu_101, convolution_128, buf72, buf417, buf73, buf416, primals_387, 1024, grid=grid(1024), stream=stream0)
        del convolution_128
        del primals_387
        del relu_101
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf421 = aten.convolution_backward(buf418, mean_24, primals_385, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_24
        del primals_385
        buf422 = buf421[0]
        buf424 = buf388; del buf388  # reuse
        buf425 = empty((512, ), device='cuda', dtype=torch.float32)
        buf427 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_100, buf408, div_24, buf422, convolution_127, unsqueeze_996, squeeze_310, buf424, buf425, buf427, 512, 2048, grid=grid(512), stream=stream0)
        buf426 = buf391; del buf391  # reuse
        buf428 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf428, relu_100, buf408, div_24, buf422, convolution_127, unsqueeze_996, buf425, squeeze_310, buf424, primals_383, 1048576, grid=grid(1048576), stream=stream0)
        del buf408
        del buf422
        del convolution_127
        del div_24
        del primals_383
        del relu_100
        del squeeze_310
        del unsqueeze_996
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf429 = aten.convolution_backward(buf428, relu_99, primals_382, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_382
        buf430 = buf429[0]
        buf432 = buf396; del buf396  # reuse
        buf433 = empty((256, ), device='cuda', dtype=torch.float32)
        buf434 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_99, buf430, convolution_126, unsqueeze_1008, squeeze_307, buf432, buf433, buf434, 256, 2048, grid=grid(256), stream=stream0)
        buf435 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf435, relu_99, convolution_126, unsqueeze_1008, buf433, squeeze_307, buf432, primals_380, 524288, grid=grid(524288), stream=stream0)
        del convolution_126
        del primals_380
        del relu_99
        del squeeze_307
        del unsqueeze_1008
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf436 = aten.convolution_backward(buf435, relu_98, primals_379, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf435
        del primals_379
        buf437 = buf436[0]
        buf439 = buf404; del buf404  # reuse
        buf440 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf442 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_98, buf402, buf437, convolution_125, unsqueeze_1020, squeeze_304, buf439, buf440, buf442, 1024, 2048, grid=grid(1024), stream=stream0)
        buf441 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_98, buf402, buf437, convolution_125, unsqueeze_1020, buf440, squeeze_304, buf439, primals_377, buf441, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_125
        del primals_377
        del squeeze_304
        del unsqueeze_1020
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf443 = aten.convolution_backward(buf441, sum_72, primals_376, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf441
        del primals_376
        del sum_72
        buf444 = buf443[0]
        buf446 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf444, relu_96, buf446, 4096, 256, grid=grid(4096), stream=stream0)
        buf447 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf446, div_23, buf447, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf449 = aten.convolution_backward(reinterpret_tensor(buf447, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_97, primals_374, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_374
        buf450 = buf449[0]
        buf70 = buf73; del buf73  # reuse
        buf69 = buf72; del buf72  # reuse
        buf452 = buf417; del buf417  # reuse
        buf453 = empty((128, ), device='cuda', dtype=torch.float32)
        buf455 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_118], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_123, relu_97, buf450, buf70, buf69, buf452, buf453, buf455, 128, 8, grid=grid(128), stream=stream0)
        buf454 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf454, relu_97, convolution_123, buf69, buf453, buf70, buf452, primals_372, 1024, grid=grid(1024), stream=stream0)
        del convolution_123
        del primals_372
        del relu_97
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf457 = aten.convolution_backward(buf454, mean_23, primals_370, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_23
        del primals_370
        buf458 = buf457[0]
        buf460 = buf425; del buf425  # reuse
        buf461 = empty((512, ), device='cuda', dtype=torch.float32)
        buf463 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_96, buf444, div_23, buf458, convolution_122, unsqueeze_1046, squeeze_298, buf460, buf461, buf463, 512, 2048, grid=grid(512), stream=stream0)
        buf462 = buf428; del buf428  # reuse
        buf464 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf464, relu_96, buf444, div_23, buf458, convolution_122, unsqueeze_1046, buf461, squeeze_298, buf460, primals_368, 1048576, grid=grid(1048576), stream=stream0)
        del buf444
        del buf458
        del convolution_122
        del div_23
        del primals_368
        del relu_96
        del squeeze_298
        del unsqueeze_1046
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf465 = aten.convolution_backward(buf464, relu_95, primals_367, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_367
        buf466 = buf465[0]
        buf468 = buf433; del buf433  # reuse
        buf469 = empty((256, ), device='cuda', dtype=torch.float32)
        buf470 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_95, buf466, convolution_121, unsqueeze_1058, squeeze_295, buf468, buf469, buf470, 256, 2048, grid=grid(256), stream=stream0)
        buf471 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf471, relu_95, convolution_121, unsqueeze_1058, buf469, squeeze_295, buf468, primals_365, 524288, grid=grid(524288), stream=stream0)
        del convolution_121
        del primals_365
        del relu_95
        del squeeze_295
        del unsqueeze_1058
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf472 = aten.convolution_backward(buf471, relu_94, primals_364, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf471
        del primals_364
        buf473 = buf472[0]
        buf475 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf475, relu_94, relu_98, buf437, buf473, 2097152, grid=grid(2097152), stream=stream0)
        del buf437
        del relu_94
        del relu_98
        buf476 = buf440; del buf440  # reuse
        buf477 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf478 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf475, convolution_120, unsqueeze_1070, squeeze_292, buf476, buf477, buf478, 1024, 2048, grid=grid(1024), stream=stream0)
        buf479 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf475, convolution_120, unsqueeze_1070, buf477, squeeze_292, buf476, primals_362, buf479, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_120
        del primals_362
        del squeeze_292
        del unsqueeze_1070
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf480 = aten.convolution_backward(buf479, sum_69, primals_361, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_361
        del sum_69
        buf481 = buf480[0]
        buf483 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf481, relu_92, buf483, 4096, 256, grid=grid(4096), stream=stream0)
        buf484 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf483, div_22, buf484, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf486 = aten.convolution_backward(reinterpret_tensor(buf484, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_93, primals_359, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_359
        buf487 = buf486[0]
        buf67 = buf70; del buf70  # reuse
        buf66 = buf69; del buf69  # reuse
        buf489 = buf453; del buf453  # reuse
        buf490 = empty((128, ), device='cuda', dtype=torch.float32)
        buf492 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_113], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_118, relu_93, buf487, buf67, buf66, buf489, buf490, buf492, 128, 8, grid=grid(128), stream=stream0)
        buf491 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf491, relu_93, convolution_118, buf66, buf490, buf67, buf489, primals_357, 1024, grid=grid(1024), stream=stream0)
        del convolution_118
        del primals_357
        del relu_93
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf494 = aten.convolution_backward(buf491, mean_22, primals_355, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_22
        del primals_355
        buf495 = buf494[0]
        buf497 = buf461; del buf461  # reuse
        buf498 = empty((512, ), device='cuda', dtype=torch.float32)
        buf500 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_92, buf481, div_22, buf495, convolution_117, unsqueeze_1096, squeeze_286, buf497, buf498, buf500, 512, 2048, grid=grid(512), stream=stream0)
        buf499 = buf464; del buf464  # reuse
        buf501 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf501, relu_92, buf481, div_22, buf495, convolution_117, unsqueeze_1096, buf498, squeeze_286, buf497, primals_353, 1048576, grid=grid(1048576), stream=stream0)
        del buf481
        del buf495
        del convolution_117
        del div_22
        del primals_353
        del relu_92
        del squeeze_286
        del unsqueeze_1096
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf502 = aten.convolution_backward(buf501, relu_91, primals_352, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_352
        buf503 = buf502[0]
        buf505 = buf469; del buf469  # reuse
        buf506 = empty((256, ), device='cuda', dtype=torch.float32)
        buf507 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_91, buf503, convolution_116, unsqueeze_1108, squeeze_283, buf505, buf506, buf507, 256, 2048, grid=grid(256), stream=stream0)
        buf508 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf508, relu_91, convolution_116, unsqueeze_1108, buf506, squeeze_283, buf505, primals_350, 524288, grid=grid(524288), stream=stream0)
        del convolution_116
        del primals_350
        del relu_91
        del squeeze_283
        del unsqueeze_1108
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf509 = aten.convolution_backward(buf508, relu_90, primals_349, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf508
        del primals_349
        buf510 = buf509[0]
        buf512 = buf477; del buf477  # reuse
        buf513 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf515 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_90, buf475, buf510, convolution_115, unsqueeze_1120, squeeze_280, buf512, buf513, buf515, 1024, 2048, grid=grid(1024), stream=stream0)
        buf514 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_90, buf475, buf510, convolution_115, unsqueeze_1120, buf513, squeeze_280, buf512, primals_347, buf514, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_115
        del primals_347
        del squeeze_280
        del unsqueeze_1120
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf516 = aten.convolution_backward(buf514, sum_66, primals_346, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf514
        del primals_346
        del sum_66
        buf517 = buf516[0]
        buf519 = buf483; del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf517, relu_88, buf519, 4096, 256, grid=grid(4096), stream=stream0)
        buf520 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf519, div_21, buf520, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf522 = aten.convolution_backward(reinterpret_tensor(buf520, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_89, primals_344, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_344
        buf523 = buf522[0]
        buf64 = buf67; del buf67  # reuse
        buf63 = buf66; del buf66  # reuse
        buf525 = buf490; del buf490  # reuse
        buf526 = empty((128, ), device='cuda', dtype=torch.float32)
        buf528 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_113, relu_89, buf523, buf64, buf63, buf525, buf526, buf528, 128, 8, grid=grid(128), stream=stream0)
        buf527 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf527, relu_89, convolution_113, buf63, buf526, buf64, buf525, primals_342, 1024, grid=grid(1024), stream=stream0)
        del convolution_113
        del primals_342
        del relu_89
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf530 = aten.convolution_backward(buf527, mean_21, primals_340, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_21
        del primals_340
        buf531 = buf530[0]
        buf533 = buf498; del buf498  # reuse
        buf534 = empty((512, ), device='cuda', dtype=torch.float32)
        buf536 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_88, buf517, div_21, buf531, convolution_112, unsqueeze_1146, squeeze_274, buf533, buf534, buf536, 512, 2048, grid=grid(512), stream=stream0)
        buf535 = buf501; del buf501  # reuse
        buf537 = buf535; del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf537, relu_88, buf517, div_21, buf531, convolution_112, unsqueeze_1146, buf534, squeeze_274, buf533, primals_338, 1048576, grid=grid(1048576), stream=stream0)
        del buf517
        del buf531
        del convolution_112
        del div_21
        del primals_338
        del relu_88
        del squeeze_274
        del unsqueeze_1146
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf538 = aten.convolution_backward(buf537, relu_87, primals_337, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_337
        buf539 = buf538[0]
        buf541 = buf506; del buf506  # reuse
        buf542 = empty((256, ), device='cuda', dtype=torch.float32)
        buf543 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_87, buf539, convolution_111, unsqueeze_1158, squeeze_271, buf541, buf542, buf543, 256, 2048, grid=grid(256), stream=stream0)
        buf544 = buf539; del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf544, relu_87, convolution_111, unsqueeze_1158, buf542, squeeze_271, buf541, primals_335, 524288, grid=grid(524288), stream=stream0)
        del convolution_111
        del primals_335
        del relu_87
        del squeeze_271
        del unsqueeze_1158
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf545 = aten.convolution_backward(buf544, relu_86, primals_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf544
        del primals_334
        buf546 = buf545[0]
        buf548 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf548, relu_86, relu_90, buf510, buf546, 2097152, grid=grid(2097152), stream=stream0)
        del buf510
        del relu_86
        del relu_90
        buf549 = buf513; del buf513  # reuse
        buf550 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf551 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf548, convolution_110, unsqueeze_1170, squeeze_268, buf549, buf550, buf551, 1024, 2048, grid=grid(1024), stream=stream0)
        buf552 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf548, convolution_110, unsqueeze_1170, buf550, squeeze_268, buf549, primals_332, buf552, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_110
        del primals_332
        del squeeze_268
        del unsqueeze_1170
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf553 = aten.convolution_backward(buf552, sum_63, primals_331, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_331
        del sum_63
        buf554 = buf553[0]
        buf556 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf554, relu_84, buf556, 4096, 256, grid=grid(4096), stream=stream0)
        buf557 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf556, div_20, buf557, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf559 = aten.convolution_backward(reinterpret_tensor(buf557, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_85, primals_329, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_329
        buf560 = buf559[0]
        buf61 = buf64; del buf64  # reuse
        buf60 = buf63; del buf63  # reuse
        buf562 = buf526; del buf526  # reuse
        buf563 = empty((128, ), device='cuda', dtype=torch.float32)
        buf565 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_103], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_108, relu_85, buf560, buf61, buf60, buf562, buf563, buf565, 128, 8, grid=grid(128), stream=stream0)
        buf564 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf564, relu_85, convolution_108, buf60, buf563, buf61, buf562, primals_327, 1024, grid=grid(1024), stream=stream0)
        del convolution_108
        del primals_327
        del relu_85
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf567 = aten.convolution_backward(buf564, mean_20, primals_325, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_20
        del primals_325
        buf568 = buf567[0]
        buf570 = buf534; del buf534  # reuse
        buf571 = empty((512, ), device='cuda', dtype=torch.float32)
        buf573 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_84, buf554, div_20, buf568, convolution_107, unsqueeze_1196, squeeze_262, buf570, buf571, buf573, 512, 2048, grid=grid(512), stream=stream0)
        buf572 = buf537; del buf537  # reuse
        buf574 = buf572; del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf574, relu_84, buf554, div_20, buf568, convolution_107, unsqueeze_1196, buf571, squeeze_262, buf570, primals_323, 1048576, grid=grid(1048576), stream=stream0)
        del buf554
        del buf568
        del convolution_107
        del div_20
        del primals_323
        del relu_84
        del squeeze_262
        del unsqueeze_1196
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf575 = aten.convolution_backward(buf574, relu_83, primals_322, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_322
        buf576 = buf575[0]
        buf578 = buf542; del buf542  # reuse
        buf579 = empty((256, ), device='cuda', dtype=torch.float32)
        buf580 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_83, buf576, convolution_106, unsqueeze_1208, squeeze_259, buf578, buf579, buf580, 256, 2048, grid=grid(256), stream=stream0)
        buf581 = buf576; del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf581, relu_83, convolution_106, unsqueeze_1208, buf579, squeeze_259, buf578, primals_320, 524288, grid=grid(524288), stream=stream0)
        del convolution_106
        del primals_320
        del relu_83
        del squeeze_259
        del unsqueeze_1208
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf582 = aten.convolution_backward(buf581, relu_82, primals_319, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf581
        del primals_319
        buf583 = buf582[0]
        buf585 = buf550; del buf550  # reuse
        buf586 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf588 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_82, buf548, buf583, convolution_105, unsqueeze_1220, squeeze_256, buf585, buf586, buf588, 1024, 2048, grid=grid(1024), stream=stream0)
        buf587 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_82, buf548, buf583, convolution_105, unsqueeze_1220, buf586, squeeze_256, buf585, primals_317, buf587, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_105
        del primals_317
        del squeeze_256
        del unsqueeze_1220
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf589 = aten.convolution_backward(buf587, sum_60, primals_316, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf587
        del primals_316
        del sum_60
        buf590 = buf589[0]
        buf592 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf590, relu_80, buf592, 4096, 256, grid=grid(4096), stream=stream0)
        buf593 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf592, div_19, buf593, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf595 = aten.convolution_backward(reinterpret_tensor(buf593, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_81, primals_314, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_314
        buf596 = buf595[0]
        buf58 = buf61; del buf61  # reuse
        buf57 = buf60; del buf60  # reuse
        buf598 = buf563; del buf563  # reuse
        buf599 = empty((128, ), device='cuda', dtype=torch.float32)
        buf601 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_103, relu_81, buf596, buf58, buf57, buf598, buf599, buf601, 128, 8, grid=grid(128), stream=stream0)
        buf600 = buf596; del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf600, relu_81, convolution_103, buf57, buf599, buf58, buf598, primals_312, 1024, grid=grid(1024), stream=stream0)
        del convolution_103
        del primals_312
        del relu_81
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf603 = aten.convolution_backward(buf600, mean_19, primals_310, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_19
        del primals_310
        buf604 = buf603[0]
        buf606 = buf571; del buf571  # reuse
        buf607 = empty((512, ), device='cuda', dtype=torch.float32)
        buf609 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_80, buf590, div_19, buf604, convolution_102, unsqueeze_1246, squeeze_250, buf606, buf607, buf609, 512, 2048, grid=grid(512), stream=stream0)
        buf608 = buf574; del buf574  # reuse
        buf610 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf610, relu_80, buf590, div_19, buf604, convolution_102, unsqueeze_1246, buf607, squeeze_250, buf606, primals_308, 1048576, grid=grid(1048576), stream=stream0)
        del buf590
        del buf604
        del convolution_102
        del div_19
        del primals_308
        del relu_80
        del squeeze_250
        del unsqueeze_1246
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf611 = aten.convolution_backward(buf610, relu_79, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_307
        buf612 = buf611[0]
        buf614 = buf579; del buf579  # reuse
        buf615 = empty((256, ), device='cuda', dtype=torch.float32)
        buf616 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_79, buf612, convolution_101, unsqueeze_1258, squeeze_247, buf614, buf615, buf616, 256, 2048, grid=grid(256), stream=stream0)
        buf617 = buf612; del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf617, relu_79, convolution_101, unsqueeze_1258, buf615, squeeze_247, buf614, primals_305, 524288, grid=grid(524288), stream=stream0)
        del convolution_101
        del primals_305
        del relu_79
        del squeeze_247
        del unsqueeze_1258
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf618 = aten.convolution_backward(buf617, relu_78, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf617
        del primals_304
        buf619 = buf618[0]
        buf621 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf621, relu_78, relu_82, buf583, buf619, 2097152, grid=grid(2097152), stream=stream0)
        del buf583
        del relu_78
        del relu_82
        buf622 = buf586; del buf586  # reuse
        buf623 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf624 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf621, convolution_100, unsqueeze_1270, squeeze_244, buf622, buf623, buf624, 1024, 2048, grid=grid(1024), stream=stream0)
        buf625 = buf619; del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf621, convolution_100, unsqueeze_1270, buf623, squeeze_244, buf622, primals_302, buf625, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_100
        del primals_302
        del squeeze_244
        del unsqueeze_1270
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf626 = aten.convolution_backward(buf625, sum_57, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_301
        del sum_57
        buf627 = buf626[0]
        buf629 = buf592; del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf627, relu_76, buf629, 4096, 256, grid=grid(4096), stream=stream0)
        buf630 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf629, div_18, buf630, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf632 = aten.convolution_backward(reinterpret_tensor(buf630, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_77, primals_299, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_299
        buf633 = buf632[0]
        buf55 = reinterpret_tensor(buf599, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf599  # reuse
        buf54 = reinterpret_tensor(buf58, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf58  # reuse
        buf635 = reinterpret_tensor(buf57, (128, ), (1, ), 0); del buf57  # reuse
        buf636 = empty((128, ), device='cuda', dtype=torch.float32)
        buf638 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_93], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_98, relu_77, buf633, buf55, buf54, buf635, buf636, buf638, 128, 8, grid=grid(128), stream=stream0)
        buf637 = buf633; del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf637, relu_77, convolution_98, buf54, buf636, buf55, buf635, primals_297, 1024, grid=grid(1024), stream=stream0)
        del convolution_98
        del primals_297
        del relu_77
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf640 = aten.convolution_backward(buf637, mean_18, primals_295, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_18
        del primals_295
        buf641 = buf640[0]
        buf643 = buf607; del buf607  # reuse
        buf644 = empty((512, ), device='cuda', dtype=torch.float32)
        buf646 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_76, buf627, div_18, buf641, convolution_97, unsqueeze_1296, squeeze_238, buf643, buf644, buf646, 512, 2048, grid=grid(512), stream=stream0)
        buf645 = buf610; del buf610  # reuse
        buf647 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf647, relu_76, buf627, div_18, buf641, convolution_97, unsqueeze_1296, buf644, squeeze_238, buf643, primals_293, 1048576, grid=grid(1048576), stream=stream0)
        del buf627
        del buf641
        del convolution_97
        del div_18
        del primals_293
        del relu_76
        del squeeze_238
        del unsqueeze_1296
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf648 = aten.convolution_backward(buf647, relu_75, primals_292, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_292
        buf649 = buf648[0]
        buf651 = buf615; del buf615  # reuse
        buf652 = empty((256, ), device='cuda', dtype=torch.float32)
        buf653 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_75, buf649, convolution_96, unsqueeze_1308, squeeze_235, buf651, buf652, buf653, 256, 2048, grid=grid(256), stream=stream0)
        buf654 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf654, relu_75, convolution_96, unsqueeze_1308, buf652, squeeze_235, buf651, primals_290, 524288, grid=grid(524288), stream=stream0)
        del convolution_96
        del primals_290
        del relu_75
        del squeeze_235
        del unsqueeze_1308
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf655 = aten.convolution_backward(buf654, relu_74, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf654
        del primals_289
        buf656 = buf655[0]
        buf658 = buf623; del buf623  # reuse
        buf659 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf661 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_74, buf621, buf656, convolution_95, unsqueeze_1320, squeeze_232, buf658, buf659, buf661, 1024, 2048, grid=grid(1024), stream=stream0)
        buf660 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_74, buf621, buf656, convolution_95, unsqueeze_1320, buf659, squeeze_232, buf658, primals_287, buf660, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_95
        del primals_287
        del squeeze_232
        del unsqueeze_1320
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf662 = aten.convolution_backward(buf660, sum_54, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf660
        del primals_286
        del sum_54
        buf663 = buf662[0]
        buf665 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf663, relu_72, buf665, 4096, 256, grid=grid(4096), stream=stream0)
        buf666 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf665, div_17, buf666, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf668 = aten.convolution_backward(reinterpret_tensor(buf666, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_73, primals_284, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_284
        buf669 = buf668[0]
        buf52 = reinterpret_tensor(buf636, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf636  # reuse
        buf51 = reinterpret_tensor(buf55, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf55  # reuse
        buf671 = reinterpret_tensor(buf54, (128, ), (1, ), 0); del buf54  # reuse
        buf672 = empty((128, ), device='cuda', dtype=torch.float32)
        buf674 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_93, relu_73, buf669, buf52, buf51, buf671, buf672, buf674, 128, 8, grid=grid(128), stream=stream0)
        buf673 = buf669; del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf673, relu_73, convolution_93, buf51, buf672, buf52, buf671, primals_282, 1024, grid=grid(1024), stream=stream0)
        del convolution_93
        del primals_282
        del relu_73
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf676 = aten.convolution_backward(buf673, mean_17, primals_280, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_17
        del primals_280
        buf677 = buf676[0]
        buf679 = buf644; del buf644  # reuse
        buf680 = empty((512, ), device='cuda', dtype=torch.float32)
        buf682 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_72, buf663, div_17, buf677, convolution_92, unsqueeze_1346, squeeze_226, buf679, buf680, buf682, 512, 2048, grid=grid(512), stream=stream0)
        buf681 = buf647; del buf647  # reuse
        buf683 = buf681; del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf683, relu_72, buf663, div_17, buf677, convolution_92, unsqueeze_1346, buf680, squeeze_226, buf679, primals_278, 1048576, grid=grid(1048576), stream=stream0)
        del buf663
        del buf677
        del convolution_92
        del div_17
        del primals_278
        del relu_72
        del squeeze_226
        del unsqueeze_1346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf684 = aten.convolution_backward(buf683, relu_71, primals_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_277
        buf685 = buf684[0]
        buf687 = buf652; del buf652  # reuse
        buf688 = empty((256, ), device='cuda', dtype=torch.float32)
        buf689 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_71, buf685, convolution_91, unsqueeze_1358, squeeze_223, buf687, buf688, buf689, 256, 2048, grid=grid(256), stream=stream0)
        buf690 = buf685; del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf690, relu_71, convolution_91, unsqueeze_1358, buf688, squeeze_223, buf687, primals_275, 524288, grid=grid(524288), stream=stream0)
        del convolution_91
        del primals_275
        del relu_71
        del squeeze_223
        del unsqueeze_1358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf691 = aten.convolution_backward(buf690, relu_70, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf690
        del primals_274
        buf692 = buf691[0]
        buf694 = buf621; del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf694, relu_70, relu_74, buf656, buf692, 2097152, grid=grid(2097152), stream=stream0)
        del buf656
        del relu_70
        del relu_74
        buf695 = buf659; del buf659  # reuse
        buf696 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf697 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf694, convolution_90, unsqueeze_1370, squeeze_220, buf695, buf696, buf697, 1024, 2048, grid=grid(1024), stream=stream0)
        buf698 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf694, convolution_90, unsqueeze_1370, buf696, squeeze_220, buf695, primals_272, buf698, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_90
        del primals_272
        del squeeze_220
        del unsqueeze_1370
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf699 = aten.convolution_backward(buf698, sum_51, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_271
        del sum_51
        buf700 = buf699[0]
        buf702 = buf665; del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf700, relu_68, buf702, 4096, 256, grid=grid(4096), stream=stream0)
        buf703 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf702, div_16, buf703, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf705 = aten.convolution_backward(reinterpret_tensor(buf703, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_69, primals_269, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_269
        buf706 = buf705[0]
        buf49 = reinterpret_tensor(buf672, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf672  # reuse
        buf48 = reinterpret_tensor(buf52, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf52  # reuse
        buf708 = reinterpret_tensor(buf51, (128, ), (1, ), 0); del buf51  # reuse
        buf709 = empty((128, ), device='cuda', dtype=torch.float32)
        buf711 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_83], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_88, relu_69, buf706, buf49, buf48, buf708, buf709, buf711, 128, 8, grid=grid(128), stream=stream0)
        buf710 = buf706; del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf710, relu_69, convolution_88, buf48, buf709, buf49, buf708, primals_267, 1024, grid=grid(1024), stream=stream0)
        del convolution_88
        del primals_267
        del relu_69
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf713 = aten.convolution_backward(buf710, mean_16, primals_265, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_16
        del primals_265
        buf714 = buf713[0]
        buf716 = buf680; del buf680  # reuse
        buf717 = empty((512, ), device='cuda', dtype=torch.float32)
        buf719 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_68, buf700, div_16, buf714, convolution_87, unsqueeze_1396, squeeze_214, buf716, buf717, buf719, 512, 2048, grid=grid(512), stream=stream0)
        buf718 = buf683; del buf683  # reuse
        buf720 = buf718; del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf720, relu_68, buf700, div_16, buf714, convolution_87, unsqueeze_1396, buf717, squeeze_214, buf716, primals_263, 1048576, grid=grid(1048576), stream=stream0)
        del buf700
        del buf714
        del convolution_87
        del div_16
        del primals_263
        del relu_68
        del squeeze_214
        del unsqueeze_1396
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf721 = aten.convolution_backward(buf720, relu_67, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_262
        buf722 = buf721[0]
        buf724 = buf688; del buf688  # reuse
        buf725 = empty((256, ), device='cuda', dtype=torch.float32)
        buf726 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_67, buf722, convolution_86, unsqueeze_1408, squeeze_211, buf724, buf725, buf726, 256, 2048, grid=grid(256), stream=stream0)
        buf727 = buf722; del buf722  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf727, relu_67, convolution_86, unsqueeze_1408, buf725, squeeze_211, buf724, primals_260, 524288, grid=grid(524288), stream=stream0)
        del convolution_86
        del primals_260
        del relu_67
        del squeeze_211
        del unsqueeze_1408
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf728 = aten.convolution_backward(buf727, relu_66, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf727
        del primals_259
        buf729 = buf728[0]
        buf731 = buf696; del buf696  # reuse
        buf732 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf734 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_66, buf694, buf729, convolution_85, unsqueeze_1420, squeeze_208, buf731, buf732, buf734, 1024, 2048, grid=grid(1024), stream=stream0)
        buf733 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_66, buf694, buf729, convolution_85, unsqueeze_1420, buf732, squeeze_208, buf731, primals_257, buf733, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_85
        del primals_257
        del squeeze_208
        del unsqueeze_1420
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf735 = aten.convolution_backward(buf733, sum_48, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf733
        del primals_256
        del sum_48
        buf736 = buf735[0]
        buf738 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf736, relu_64, buf738, 4096, 256, grid=grid(4096), stream=stream0)
        buf739 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf738, div_15, buf739, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf741 = aten.convolution_backward(reinterpret_tensor(buf739, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_65, primals_254, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_254
        buf742 = buf741[0]
        buf46 = reinterpret_tensor(buf709, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf709  # reuse
        buf45 = reinterpret_tensor(buf49, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf49  # reuse
        buf744 = reinterpret_tensor(buf48, (128, ), (1, ), 0); del buf48  # reuse
        buf745 = empty((128, ), device='cuda', dtype=torch.float32)
        buf747 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_83, relu_65, buf742, buf46, buf45, buf744, buf745, buf747, 128, 8, grid=grid(128), stream=stream0)
        buf746 = buf742; del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf746, relu_65, convolution_83, buf45, buf745, buf46, buf744, primals_252, 1024, grid=grid(1024), stream=stream0)
        del convolution_83
        del primals_252
        del relu_65
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf749 = aten.convolution_backward(buf746, mean_15, primals_250, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_15
        del primals_250
        buf750 = buf749[0]
        buf752 = buf717; del buf717  # reuse
        buf753 = empty((512, ), device='cuda', dtype=torch.float32)
        buf755 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_64, buf736, div_15, buf750, convolution_82, unsqueeze_1446, squeeze_202, buf752, buf753, buf755, 512, 2048, grid=grid(512), stream=stream0)
        buf754 = buf720; del buf720  # reuse
        buf756 = buf754; del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf756, relu_64, buf736, div_15, buf750, convolution_82, unsqueeze_1446, buf753, squeeze_202, buf752, primals_248, 1048576, grid=grid(1048576), stream=stream0)
        del buf736
        del buf750
        del convolution_82
        del div_15
        del primals_248
        del relu_64
        del squeeze_202
        del unsqueeze_1446
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf757 = aten.convolution_backward(buf756, relu_63, primals_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_247
        buf758 = buf757[0]
        buf760 = buf725; del buf725  # reuse
        buf761 = empty((256, ), device='cuda', dtype=torch.float32)
        buf762 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_63, buf758, convolution_81, unsqueeze_1458, squeeze_199, buf760, buf761, buf762, 256, 2048, grid=grid(256), stream=stream0)
        buf763 = buf758; del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf763, relu_63, convolution_81, unsqueeze_1458, buf761, squeeze_199, buf760, primals_245, 524288, grid=grid(524288), stream=stream0)
        del convolution_81
        del primals_245
        del relu_63
        del squeeze_199
        del unsqueeze_1458
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf764 = aten.convolution_backward(buf763, relu_62, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf763
        del primals_244
        buf765 = buf764[0]
        buf767 = buf694; del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf767, relu_62, relu_66, buf729, buf765, 2097152, grid=grid(2097152), stream=stream0)
        del buf729
        del relu_62
        del relu_66
        buf768 = buf732; del buf732  # reuse
        buf769 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf770 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf767, convolution_80, unsqueeze_1470, squeeze_196, buf768, buf769, buf770, 1024, 2048, grid=grid(1024), stream=stream0)
        buf771 = buf765; del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf767, convolution_80, unsqueeze_1470, buf769, squeeze_196, buf768, primals_242, buf771, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_80
        del primals_242
        del squeeze_196
        del unsqueeze_1470
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf772 = aten.convolution_backward(buf771, sum_45, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_241
        del sum_45
        buf773 = buf772[0]
        buf775 = buf738; del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf773, relu_60, buf775, 4096, 256, grid=grid(4096), stream=stream0)
        buf776 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf775, div_14, buf776, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf778 = aten.convolution_backward(reinterpret_tensor(buf776, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_61, primals_239, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_239
        buf779 = buf778[0]
        buf43 = reinterpret_tensor(buf745, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf745  # reuse
        buf42 = reinterpret_tensor(buf46, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf46  # reuse
        buf781 = reinterpret_tensor(buf45, (128, ), (1, ), 0); del buf45  # reuse
        buf782 = empty((128, ), device='cuda', dtype=torch.float32)
        buf784 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_73], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_78, relu_61, buf779, buf43, buf42, buf781, buf782, buf784, 128, 8, grid=grid(128), stream=stream0)
        buf783 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf783, relu_61, convolution_78, buf42, buf782, buf43, buf781, primals_237, 1024, grid=grid(1024), stream=stream0)
        del convolution_78
        del primals_237
        del relu_61
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf786 = aten.convolution_backward(buf783, mean_14, primals_235, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_14
        del primals_235
        buf787 = buf786[0]
        buf789 = buf753; del buf753  # reuse
        buf790 = empty((512, ), device='cuda', dtype=torch.float32)
        buf792 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_60, buf773, div_14, buf787, convolution_77, unsqueeze_1496, squeeze_190, buf789, buf790, buf792, 512, 2048, grid=grid(512), stream=stream0)
        buf791 = buf756; del buf756  # reuse
        buf793 = buf791; del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf793, relu_60, buf773, div_14, buf787, convolution_77, unsqueeze_1496, buf790, squeeze_190, buf789, primals_233, 1048576, grid=grid(1048576), stream=stream0)
        del buf773
        del buf787
        del convolution_77
        del div_14
        del primals_233
        del relu_60
        del squeeze_190
        del unsqueeze_1496
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf794 = aten.convolution_backward(buf793, relu_59, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_232
        buf795 = buf794[0]
        buf797 = buf761; del buf761  # reuse
        buf798 = empty((256, ), device='cuda', dtype=torch.float32)
        buf799 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_59, buf795, convolution_76, unsqueeze_1508, squeeze_187, buf797, buf798, buf799, 256, 2048, grid=grid(256), stream=stream0)
        buf800 = buf795; del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf800, relu_59, convolution_76, unsqueeze_1508, buf798, squeeze_187, buf797, primals_230, 524288, grid=grid(524288), stream=stream0)
        del convolution_76
        del primals_230
        del relu_59
        del squeeze_187
        del unsqueeze_1508
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf801 = aten.convolution_backward(buf800, relu_58, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf800
        del primals_229
        buf802 = buf801[0]
        buf804 = buf769; del buf769  # reuse
        buf805 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf807 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_58, buf767, buf802, convolution_75, unsqueeze_1520, squeeze_184, buf804, buf805, buf807, 1024, 2048, grid=grid(1024), stream=stream0)
        buf806 = buf771; del buf771  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_58, buf767, buf802, convolution_75, unsqueeze_1520, buf805, squeeze_184, buf804, primals_227, buf806, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_75
        del primals_227
        del squeeze_184
        del unsqueeze_1520
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf808 = aten.convolution_backward(buf806, sum_42, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf806
        del primals_226
        del sum_42
        buf809 = buf808[0]
        buf811 = buf775; del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf809, relu_56, buf811, 4096, 256, grid=grid(4096), stream=stream0)
        buf812 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf811, div_13, buf812, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf814 = aten.convolution_backward(reinterpret_tensor(buf812, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_57, primals_224, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_224
        buf815 = buf814[0]
        buf40 = reinterpret_tensor(buf782, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf782  # reuse
        buf39 = reinterpret_tensor(buf43, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf43  # reuse
        buf817 = reinterpret_tensor(buf42, (128, ), (1, ), 0); del buf42  # reuse
        buf818 = empty((128, ), device='cuda', dtype=torch.float32)
        buf820 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_68], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_73, relu_57, buf815, buf40, buf39, buf817, buf818, buf820, 128, 8, grid=grid(128), stream=stream0)
        buf819 = buf815; del buf815  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf819, relu_57, convolution_73, buf39, buf818, buf40, buf817, primals_222, 1024, grid=grid(1024), stream=stream0)
        del convolution_73
        del primals_222
        del relu_57
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf822 = aten.convolution_backward(buf819, mean_13, primals_220, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_13
        del primals_220
        buf823 = buf822[0]
        buf825 = buf790; del buf790  # reuse
        buf826 = empty((512, ), device='cuda', dtype=torch.float32)
        buf828 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_56, buf809, div_13, buf823, convolution_72, unsqueeze_1546, squeeze_178, buf825, buf826, buf828, 512, 2048, grid=grid(512), stream=stream0)
        buf827 = buf793; del buf793  # reuse
        buf829 = buf827; del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf829, relu_56, buf809, div_13, buf823, convolution_72, unsqueeze_1546, buf826, squeeze_178, buf825, primals_218, 1048576, grid=grid(1048576), stream=stream0)
        del buf809
        del buf823
        del convolution_72
        del div_13
        del primals_218
        del relu_56
        del squeeze_178
        del unsqueeze_1546
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf830 = aten.convolution_backward(buf829, relu_55, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_217
        buf831 = buf830[0]
        buf833 = buf798; del buf798  # reuse
        buf834 = empty((256, ), device='cuda', dtype=torch.float32)
        buf835 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_55, buf831, convolution_71, unsqueeze_1558, squeeze_175, buf833, buf834, buf835, 256, 2048, grid=grid(256), stream=stream0)
        buf836 = buf831; del buf831  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf836, relu_55, convolution_71, unsqueeze_1558, buf834, squeeze_175, buf833, primals_215, 524288, grid=grid(524288), stream=stream0)
        del convolution_71
        del primals_215
        del relu_55
        del squeeze_175
        del unsqueeze_1558
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf837 = aten.convolution_backward(buf836, relu_54, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf836
        del primals_214
        buf838 = buf837[0]
        buf840 = buf767; del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf840, relu_54, relu_58, buf802, buf838, 2097152, grid=grid(2097152), stream=stream0)
        del buf802
        del relu_54
        del relu_58
        buf841 = buf805; del buf805  # reuse
        buf842 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf843 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf840, convolution_70, unsqueeze_1570, squeeze_172, buf841, buf842, buf843, 1024, 2048, grid=grid(1024), stream=stream0)
        buf844 = buf838; del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf840, convolution_70, unsqueeze_1570, buf842, squeeze_172, buf841, primals_212, buf844, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_70
        del primals_212
        del squeeze_172
        del unsqueeze_1570
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf845 = aten.convolution_backward(buf844, sum_39, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_211
        del sum_39
        buf846 = buf845[0]
        buf848 = buf811; del buf811  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf846, relu_52, buf848, 4096, 256, grid=grid(4096), stream=stream0)
        buf849 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf848, div_12, buf849, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf851 = aten.convolution_backward(reinterpret_tensor(buf849, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_53, primals_209, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_209
        buf852 = buf851[0]
        buf37 = reinterpret_tensor(buf818, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf818  # reuse
        buf36 = reinterpret_tensor(buf40, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf40  # reuse
        buf854 = reinterpret_tensor(buf39, (128, ), (1, ), 0); del buf39  # reuse
        buf855 = empty((128, ), device='cuda', dtype=torch.float32)
        buf857 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_68, relu_53, buf852, buf37, buf36, buf854, buf855, buf857, 128, 8, grid=grid(128), stream=stream0)
        buf856 = buf852; del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf856, relu_53, convolution_68, buf36, buf855, buf37, buf854, primals_207, 1024, grid=grid(1024), stream=stream0)
        del convolution_68
        del primals_207
        del relu_53
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf859 = aten.convolution_backward(buf856, mean_12, primals_205, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_12
        del primals_205
        buf860 = buf859[0]
        buf862 = buf826; del buf826  # reuse
        buf863 = empty((512, ), device='cuda', dtype=torch.float32)
        buf865 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_52, buf846, div_12, buf860, convolution_67, unsqueeze_1596, squeeze_166, buf862, buf863, buf865, 512, 2048, grid=grid(512), stream=stream0)
        buf864 = buf829; del buf829  # reuse
        buf866 = buf864; del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf866, relu_52, buf846, div_12, buf860, convolution_67, unsqueeze_1596, buf863, squeeze_166, buf862, primals_203, 1048576, grid=grid(1048576), stream=stream0)
        del buf846
        del buf860
        del convolution_67
        del div_12
        del primals_203
        del relu_52
        del squeeze_166
        del unsqueeze_1596
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf867 = aten.convolution_backward(buf866, relu_51, primals_202, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_202
        buf868 = buf867[0]
        buf870 = buf834; del buf834  # reuse
        buf871 = empty((256, ), device='cuda', dtype=torch.float32)
        buf872 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_51, buf868, convolution_66, unsqueeze_1608, squeeze_163, buf870, buf871, buf872, 256, 2048, grid=grid(256), stream=stream0)
        buf873 = buf868; del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf873, relu_51, convolution_66, unsqueeze_1608, buf871, squeeze_163, buf870, primals_200, 524288, grid=grid(524288), stream=stream0)
        del convolution_66
        del primals_200
        del relu_51
        del squeeze_163
        del unsqueeze_1608
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf874 = aten.convolution_backward(buf873, relu_50, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf873
        del primals_199
        buf875 = buf874[0]
        buf877 = buf842; del buf842  # reuse
        buf878 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf880 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_50, buf840, buf875, convolution_65, unsqueeze_1620, squeeze_160, buf877, buf878, buf880, 1024, 2048, grid=grid(1024), stream=stream0)
        buf879 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_50, buf840, buf875, convolution_65, unsqueeze_1620, buf878, squeeze_160, buf877, primals_197, buf879, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_65
        del primals_197
        del squeeze_160
        del unsqueeze_1620
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf881 = aten.convolution_backward(buf879, sum_36, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf879
        del primals_196
        del sum_36
        buf882 = buf881[0]
        buf884 = buf848; del buf848  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf882, relu_48, buf884, 4096, 256, grid=grid(4096), stream=stream0)
        buf885 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf884, div_11, buf885, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf887 = aten.convolution_backward(reinterpret_tensor(buf885, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_49, primals_194, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_194
        buf888 = buf887[0]
        buf34 = reinterpret_tensor(buf855, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf855  # reuse
        buf33 = reinterpret_tensor(buf37, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf37  # reuse
        buf890 = reinterpret_tensor(buf36, (128, ), (1, ), 0); del buf36  # reuse
        buf891 = empty((128, ), device='cuda', dtype=torch.float32)
        buf893 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_63, relu_49, buf888, buf34, buf33, buf890, buf891, buf893, 128, 8, grid=grid(128), stream=stream0)
        buf892 = buf888; del buf888  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf892, relu_49, convolution_63, buf33, buf891, buf34, buf890, primals_192, 1024, grid=grid(1024), stream=stream0)
        del convolution_63
        del primals_192
        del relu_49
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf895 = aten.convolution_backward(buf892, mean_11, primals_190, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_11
        del primals_190
        buf896 = buf895[0]
        buf898 = buf863; del buf863  # reuse
        buf899 = empty((512, ), device='cuda', dtype=torch.float32)
        buf901 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_48, buf882, div_11, buf896, convolution_62, unsqueeze_1646, squeeze_154, buf898, buf899, buf901, 512, 2048, grid=grid(512), stream=stream0)
        buf900 = buf866; del buf866  # reuse
        buf902 = buf900; del buf900  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf902, relu_48, buf882, div_11, buf896, convolution_62, unsqueeze_1646, buf899, squeeze_154, buf898, primals_188, 1048576, grid=grid(1048576), stream=stream0)
        del buf882
        del convolution_62
        del div_11
        del primals_188
        del relu_48
        del squeeze_154
        del unsqueeze_1646
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf903 = aten.convolution_backward(buf902, relu_47, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_187
        buf904 = buf903[0]
        buf906 = buf871; del buf871  # reuse
        buf907 = empty((256, ), device='cuda', dtype=torch.float32)
        buf908 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_47, buf904, convolution_61, unsqueeze_1658, squeeze_151, buf906, buf907, buf908, 256, 2048, grid=grid(256), stream=stream0)
        buf909 = buf904; del buf904  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf909, relu_47, convolution_61, unsqueeze_1658, buf907, squeeze_151, buf906, primals_185, 524288, grid=grid(524288), stream=stream0)
        del convolution_61
        del primals_185
        del relu_47
        del squeeze_151
        del unsqueeze_1658
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf910 = aten.convolution_backward(buf909, relu_46, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf909
        del primals_184
        buf911 = buf910[0]
        buf913 = buf840; del buf840  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf913, relu_46, relu_50, buf875, buf911, 2097152, grid=grid(2097152), stream=stream0)
        del buf875
        del relu_46
        del relu_50
        buf914 = buf878; del buf878  # reuse
        buf915 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf916 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf913, convolution_60, unsqueeze_1670, squeeze_148, buf914, buf915, buf916, 1024, 2048, grid=grid(1024), stream=stream0)
        buf917 = buf911; del buf911  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf913, convolution_60, unsqueeze_1670, buf915, squeeze_148, buf914, primals_182, buf917, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_60
        del primals_182
        del squeeze_148
        del unsqueeze_1670
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf918 = aten.convolution_backward(buf917, sum_33, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_181
        del sum_33
        buf919 = buf918[0]
        buf921 = buf884; del buf884  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf919, relu_44, buf921, 4096, 256, grid=grid(4096), stream=stream0)
        buf922 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf921, div_10, buf922, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf924 = aten.convolution_backward(reinterpret_tensor(buf922, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_45, primals_179, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_179
        buf925 = buf924[0]
        buf31 = reinterpret_tensor(buf891, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf891  # reuse
        buf30 = reinterpret_tensor(buf34, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf34  # reuse
        buf927 = reinterpret_tensor(buf33, (128, ), (1, ), 0); del buf33  # reuse
        buf928 = empty((128, ), device='cuda', dtype=torch.float32)
        buf930 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_58, relu_45, buf925, buf31, buf30, buf927, buf928, buf930, 128, 8, grid=grid(128), stream=stream0)
        buf929 = buf925; del buf925  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf929, relu_45, convolution_58, buf30, buf928, buf31, buf927, primals_177, 1024, grid=grid(1024), stream=stream0)
        del convolution_58
        del primals_177
        del relu_45
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf932 = aten.convolution_backward(buf929, mean_10, primals_175, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_10
        del primals_175
        buf933 = buf932[0]
        buf935 = buf899; del buf899  # reuse
        buf936 = empty((512, ), device='cuda', dtype=torch.float32)
        buf938 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_44, buf919, div_10, buf933, convolution_57, unsqueeze_1696, squeeze_142, buf935, buf936, buf938, 512, 2048, grid=grid(512), stream=stream0)
        buf937 = buf902; del buf902  # reuse
        buf939 = buf937; del buf937  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf939, relu_44, buf919, div_10, buf933, convolution_57, unsqueeze_1696, buf936, squeeze_142, buf935, primals_173, 1048576, grid=grid(1048576), stream=stream0)
        del buf919
        del convolution_57
        del div_10
        del primals_173
        del relu_44
        del squeeze_142
        del unsqueeze_1696
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf940 = aten.convolution_backward(buf939, relu_43, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_172
        buf941 = buf940[0]
        buf943 = buf907; del buf907  # reuse
        buf944 = empty((256, ), device='cuda', dtype=torch.float32)
        buf945 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_43, buf941, convolution_56, unsqueeze_1708, squeeze_139, buf943, buf944, buf945, 256, 2048, grid=grid(256), stream=stream0)
        buf946 = buf941; del buf941  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf946, relu_43, convolution_56, unsqueeze_1708, buf944, squeeze_139, buf943, primals_170, 524288, grid=grid(524288), stream=stream0)
        del convolution_56
        del primals_170
        del relu_43
        del squeeze_139
        del unsqueeze_1708
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf947 = aten.convolution_backward(buf946, relu_42, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf946
        del primals_169
        buf948 = buf947[0]
        buf950 = buf915; del buf915  # reuse
        buf951 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf953 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_42, buf913, buf948, convolution_55, unsqueeze_1720, squeeze_136, buf950, buf951, buf953, 1024, 2048, grid=grid(1024), stream=stream0)
        buf952 = buf917; del buf917  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_42, buf913, buf948, convolution_55, unsqueeze_1720, buf951, squeeze_136, buf950, primals_167, buf952, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_55
        del primals_167
        del squeeze_136
        del unsqueeze_1720
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf954 = aten.convolution_backward(buf952, sum_30, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf952
        del primals_166
        del sum_30
        buf955 = buf954[0]
        buf957 = buf921; del buf921  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf955, relu_40, buf957, 4096, 256, grid=grid(4096), stream=stream0)
        buf958 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf957, div_9, buf958, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf960 = aten.convolution_backward(reinterpret_tensor(buf958, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_41, primals_164, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_164
        buf961 = buf960[0]
        buf28 = reinterpret_tensor(buf928, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf928  # reuse
        buf27 = reinterpret_tensor(buf31, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf31  # reuse
        buf963 = reinterpret_tensor(buf30, (128, ), (1, ), 0); del buf30  # reuse
        buf964 = empty((128, ), device='cuda', dtype=torch.float32)
        buf966 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_53, relu_41, buf961, buf28, buf27, buf963, buf964, buf966, 128, 8, grid=grid(128), stream=stream0)
        buf965 = buf961; del buf961  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf965, relu_41, convolution_53, buf27, buf964, buf28, buf963, primals_162, 1024, grid=grid(1024), stream=stream0)
        del convolution_53
        del primals_162
        del relu_41
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf968 = aten.convolution_backward(buf965, mean_9, primals_160, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_9
        del primals_160
        buf969 = buf968[0]
        buf971 = buf936; del buf936  # reuse
        buf972 = empty((512, ), device='cuda', dtype=torch.float32)
        buf974 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_40, buf955, div_9, buf969, convolution_52, unsqueeze_1746, squeeze_130, buf971, buf972, buf974, 512, 2048, grid=grid(512), stream=stream0)
        buf973 = buf939; del buf939  # reuse
        buf975 = buf973; del buf973  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf975, relu_40, buf955, div_9, buf969, convolution_52, unsqueeze_1746, buf972, squeeze_130, buf971, primals_158, 1048576, grid=grid(1048576), stream=stream0)
        del buf955
        del convolution_52
        del div_9
        del primals_158
        del relu_40
        del squeeze_130
        del unsqueeze_1746
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf976 = aten.convolution_backward(buf975, relu_39, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_157
        buf977 = buf976[0]
        buf979 = buf944; del buf944  # reuse
        buf980 = empty((256, ), device='cuda', dtype=torch.float32)
        buf981 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_39, buf977, convolution_51, unsqueeze_1758, squeeze_127, buf979, buf980, buf981, 256, 2048, grid=grid(256), stream=stream0)
        buf982 = buf977; del buf977  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf982, relu_39, convolution_51, unsqueeze_1758, buf980, squeeze_127, buf979, primals_155, 524288, grid=grid(524288), stream=stream0)
        del convolution_51
        del primals_155
        del relu_39
        del squeeze_127
        del unsqueeze_1758
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf983 = aten.convolution_backward(buf982, relu_38, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf982
        del primals_154
        buf984 = buf983[0]
        buf986 = buf913; del buf913  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_37.run(buf986, relu_38, relu_42, buf948, buf984, 2097152, grid=grid(2097152), stream=stream0)
        del relu_38
        del relu_42
        buf987 = buf951; del buf951  # reuse
        buf988 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf989 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf986, convolution_50, unsqueeze_1770, squeeze_124, buf987, buf988, buf989, 1024, 2048, grid=grid(1024), stream=stream0)
        buf990 = buf984; del buf984  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf986, convolution_50, unsqueeze_1770, buf988, squeeze_124, buf987, primals_152, buf990, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_50
        del primals_152
        del squeeze_124
        del unsqueeze_1770
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf991 = aten.convolution_backward(buf990, sum_27, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_151
        del sum_27
        buf992 = buf991[0]
        buf994 = buf957; del buf957  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_24.run(buf992, relu_36, buf994, 4096, 256, grid=grid(4096), stream=stream0)
        buf995 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf994, div_8, buf995, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf997 = aten.convolution_backward(reinterpret_tensor(buf995, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_37, primals_149, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_149
        buf998 = buf997[0]
        buf25 = reinterpret_tensor(buf964, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf964  # reuse
        buf24 = reinterpret_tensor(buf28, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf28  # reuse
        buf1000 = reinterpret_tensor(buf27, (128, ), (1, ), 0); del buf27  # reuse
        buf1001 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1003 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_48, relu_37, buf998, buf25, buf24, buf1000, buf1001, buf1003, 128, 8, grid=grid(128), stream=stream0)
        buf1002 = buf998; del buf998  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf1002, relu_37, convolution_48, buf24, buf1001, buf25, buf1000, primals_147, 1024, grid=grid(1024), stream=stream0)
        del convolution_48
        del primals_147
        del relu_37
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1005 = aten.convolution_backward(buf1002, mean_8, primals_145, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_8
        del primals_145
        buf1006 = buf1005[0]
        buf1008 = buf972; del buf972  # reuse
        buf1009 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1011 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_36, buf992, div_8, buf1006, convolution_47, unsqueeze_1796, squeeze_118, buf1008, buf1009, buf1011, 512, 2048, grid=grid(512), stream=stream0)
        buf1010 = buf975; del buf975  # reuse
        buf1012 = buf1010; del buf1010  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf1012, relu_36, buf992, div_8, buf1006, convolution_47, unsqueeze_1796, buf1009, squeeze_118, buf1008, primals_143, 1048576, grid=grid(1048576), stream=stream0)
        del buf992
        del convolution_47
        del div_8
        del primals_143
        del relu_36
        del squeeze_118
        del unsqueeze_1796
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1013 = aten.convolution_backward(buf1012, relu_35, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf1012
        del primals_142
        buf1014 = buf1013[0]
        buf1016 = buf980; del buf980  # reuse
        buf1017 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1018 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_30.run(relu_35, buf1014, convolution_46, unsqueeze_1808, squeeze_115, buf1016, buf1017, buf1018, 256, 2048, grid=grid(256), stream=stream0)
        buf1019 = buf1014; del buf1014  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(buf1019, relu_35, convolution_46, unsqueeze_1808, buf1017, squeeze_115, buf1016, primals_140, 524288, grid=grid(524288), stream=stream0)
        del convolution_46
        del primals_140
        del relu_35
        del squeeze_115
        del unsqueeze_1808
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1020 = aten.convolution_backward(buf1019, relu_34, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1019
        del primals_139
        buf1021 = buf1020[0]
        buf1023 = buf988; del buf988  # reuse
        buf1024 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1030 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1026 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1032 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_34, buf986, buf1021, convolution_45, unsqueeze_1820, convolution_44, unsqueeze_1832, squeeze_112, squeeze_109, buf1023, buf1024, buf1030, buf1026, buf1032, 1024, 2048, grid=grid(1024), stream=stream0)
        buf1025 = buf990; del buf990  # reuse
        buf1031 = buf948; del buf948  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_39.run(relu_34, buf986, buf1021, convolution_45, unsqueeze_1820, buf1024, squeeze_112, buf1023, primals_137, convolution_44, unsqueeze_1832, buf1030, squeeze_109, primals_134, buf1025, buf1031, 2097152, grid=grid(2097152), stream=stream0)
        del buf1021
        del buf986
        del convolution_44
        del convolution_45
        del primals_134
        del primals_137
        del relu_34
        del squeeze_109
        del squeeze_112
        del unsqueeze_1820
        del unsqueeze_1832
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1027 = aten.convolution_backward(buf1025, avg_pool2d_3, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_3
        del buf1025
        del primals_136
        buf1028 = buf1027[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1033 = aten.convolution_backward(buf1031, avg_pool2d_2, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_2
        del primals_133
        buf1034 = buf1033[0]
        buf1036 = reinterpret_tensor(buf1031, (8, 256, 32, 32), (262144, 1024, 32, 1), 0); del buf1031  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_40.run(buf1034, buf1036, 2097152, grid=grid(2097152), stream=stream0)
        del buf1034
        buf1037 = buf994; del buf994  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_41.run(buf1036, relu_32, buf1037, 4096, 1024, grid=grid(4096), stream=stream0)
        buf1038 = empty((8, 2, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_25.run(buf1037, div_7, buf1038, 4096, grid=grid(4096), stream=stream0)
        del buf1037
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1040 = aten.convolution_backward(reinterpret_tensor(buf1038, (8, 512, 1, 1), (512, 1, 0, 0), 0), relu_33, primals_131, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_131
        buf1041 = buf1040[0]
        buf22 = buf25; del buf25  # reuse
        buf21 = buf24; del buf24  # reuse
        buf1043 = buf1001; del buf1001  # reuse
        buf1044 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1046 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_26.run(convolution_42, relu_33, buf1041, buf22, buf21, buf1043, buf1044, buf1046, 128, 8, grid=grid(128), stream=stream0)
        buf1045 = buf1041; del buf1041  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_27.run(buf1045, relu_33, convolution_42, buf21, buf1044, buf22, buf1043, primals_129, 1024, grid=grid(1024), stream=stream0)
        del convolution_42
        del primals_129
        del relu_33
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1048 = aten.convolution_backward(buf1045, mean_7, primals_127, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_7
        del primals_127
        buf1049 = buf1048[0]
        buf1051 = buf1009; del buf1009  # reuse
        buf1052 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1054 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(relu_32, buf1036, div_7, buf1049, convolution_41, unsqueeze_1858, squeeze_103, buf1051, buf1052, buf1054, 512, 8192, grid=grid(512), stream=stream0)
        buf1053 = empty((8, 512, 32, 32), device='cuda', dtype=torch.float32)
        buf1055 = buf1053; del buf1053  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(buf1055, relu_32, buf1036, div_7, buf1049, convolution_41, unsqueeze_1858, buf1052, squeeze_103, buf1051, primals_125, 4194304, grid=grid(4194304), stream=stream0)
        del buf1036
        del convolution_41
        del div_7
        del primals_125
        del relu_32
        del squeeze_103
        del unsqueeze_1858
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1056 = aten.convolution_backward(buf1055, relu_31, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_124
        buf1057 = buf1056[0]
        buf1059 = buf1017; del buf1017  # reuse
        buf1060 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1061 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(relu_31, buf1057, convolution_40, unsqueeze_1870, squeeze_100, buf1059, buf1060, buf1061, 256, 8192, grid=grid(256), stream=stream0)
        buf1062 = buf1057; del buf1057  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(buf1062, relu_31, convolution_40, unsqueeze_1870, buf1060, squeeze_100, buf1059, primals_122, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_40
        del primals_122
        del relu_31
        del squeeze_100
        del unsqueeze_1870
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1063 = aten.convolution_backward(buf1062, relu_30, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_121
        buf1064 = buf1063[0]
        buf1066 = buf1052; del buf1052  # reuse
        buf1067 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1069 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_46.run(relu_30, buf1028, buf1064, convolution_39, unsqueeze_1882, squeeze_97, buf1066, buf1067, buf1069, 512, 8192, grid=grid(512), stream=stream0)
        buf1068 = buf1055; del buf1055  # reuse
        buf1070 = buf1068; del buf1068  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_47.run(buf1070, relu_30, buf1028, buf1064, convolution_39, unsqueeze_1882, buf1067, squeeze_97, buf1066, primals_119, 4194304, grid=grid(4194304), stream=stream0)
        del convolution_39
        del primals_119
        del squeeze_97
        del unsqueeze_1882
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1071 = aten.convolution_backward(buf1070, sum_21, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1070
        del primals_118
        del sum_21
        buf1072 = buf1071[0]
        buf1074 = reinterpret_tensor(buf1049, (8, 2, 128, 1, 1), (256, 128, 1, 1, 1), 0); del buf1049  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_48.run(buf1072, relu_28, buf1074, 2048, 1024, grid=grid(2048), stream=stream0)
        buf1075 = reinterpret_tensor(buf1006, (8, 2, 1, 128), (256, 128, 128, 1), 0); del buf1006  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_49.run(buf1074, div_6, buf1075, 2048, grid=grid(2048), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1077 = aten.convolution_backward(reinterpret_tensor(buf1075, (8, 256, 1, 1), (256, 1, 0, 0), 0), relu_29, primals_116, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_116
        buf1078 = buf1077[0]
        buf19 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf18 = empty((1, 64, 1, 1), device='cuda', dtype=torch.float32)
        buf1080 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1081 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1083 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_50.run(convolution_37, relu_29, buf1078, buf19, buf18, buf1080, buf1081, buf1083, 64, 8, grid=grid(64), stream=stream0)
        buf1082 = buf1078; del buf1078  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_51.run(buf1082, relu_29, convolution_37, buf18, buf1081, buf19, buf1080, primals_114, 512, grid=grid(512), stream=stream0)
        del convolution_37
        del primals_114
        del relu_29
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1085 = aten.convolution_backward(buf1082, mean_6, primals_112, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_6
        del primals_112
        buf1086 = buf1085[0]
        buf1088 = buf1060; del buf1060  # reuse
        buf1089 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1091 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_52.run(relu_28, buf1072, div_6, buf1086, convolution_36, unsqueeze_1908, squeeze_91, buf1088, buf1089, buf1091, 256, 8192, grid=grid(256), stream=stream0)
        buf1090 = buf1062; del buf1062  # reuse
        buf1092 = buf1090; del buf1090  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53.run(buf1092, relu_28, buf1072, div_6, buf1086, convolution_36, unsqueeze_1908, buf1089, squeeze_91, buf1088, primals_110, 2097152, grid=grid(2097152), stream=stream0)
        del buf1072
        del convolution_36
        del div_6
        del primals_110
        del relu_28
        del squeeze_91
        del unsqueeze_1908
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1093 = aten.convolution_backward(buf1092, relu_27, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_109
        buf1094 = buf1093[0]
        buf1096 = reinterpret_tensor(buf22, (128, ), (1, ), 0); del buf22  # reuse
        buf1097 = reinterpret_tensor(buf21, (128, ), (1, ), 0); del buf21  # reuse
        buf1098 = buf1044; del buf1044  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(relu_27, buf1094, convolution_35, unsqueeze_1920, squeeze_88, buf1096, buf1097, buf1098, 128, 8192, grid=grid(128), stream=stream0)
        buf1099 = buf1094; del buf1094  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55.run(buf1099, relu_27, convolution_35, unsqueeze_1920, buf1097, squeeze_88, buf1096, primals_107, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_35
        del primals_107
        del relu_27
        del squeeze_88
        del unsqueeze_1920
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1100 = aten.convolution_backward(buf1099, relu_26, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1099
        del primals_106
        buf1101 = buf1100[0]
        buf1103 = buf1064; del buf1064  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_threshold_backward_56.run(buf1103, relu_26, relu_30, buf1028, buf1101, 4194304, grid=grid(4194304), stream=stream0)
        del buf1028
        del relu_26
        del relu_30
        buf1104 = buf1067; del buf1067  # reuse
        buf1105 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1106 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf1103, convolution_34, unsqueeze_1932, squeeze_85, buf1104, buf1105, buf1106, 512, 8192, grid=grid(512), stream=stream0)
        buf1107 = buf1101; del buf1101  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_58.run(buf1103, convolution_34, unsqueeze_1932, buf1105, squeeze_85, buf1104, primals_104, buf1107, 4194304, grid=grid(4194304), stream=stream0)
        del convolution_34
        del primals_104
        del squeeze_85
        del unsqueeze_1932
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1108 = aten.convolution_backward(buf1107, sum_18, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_103
        del sum_18
        buf1109 = buf1108[0]
        buf1111 = buf1074; del buf1074  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_48.run(buf1109, relu_24, buf1111, 2048, 1024, grid=grid(2048), stream=stream0)
        buf1112 = reinterpret_tensor(buf969, (8, 2, 1, 128), (256, 128, 128, 1), 0); del buf969  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_49.run(buf1111, div_5, buf1112, 2048, grid=grid(2048), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1114 = aten.convolution_backward(reinterpret_tensor(buf1112, (8, 256, 1, 1), (256, 1, 0, 0), 0), relu_25, primals_101, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_101
        buf1115 = buf1114[0]
        buf16 = buf19; del buf19  # reuse
        buf15 = buf18; del buf18  # reuse
        buf1117 = buf1081; del buf1081  # reuse
        buf1118 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1120 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_50.run(convolution_32, relu_25, buf1115, buf16, buf15, buf1117, buf1118, buf1120, 64, 8, grid=grid(64), stream=stream0)
        buf1119 = buf1115; del buf1115  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_51.run(buf1119, relu_25, convolution_32, buf15, buf1118, buf16, buf1117, primals_99, 512, grid=grid(512), stream=stream0)
        del convolution_32
        del primals_99
        del relu_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1122 = aten.convolution_backward(buf1119, mean_5, primals_97, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_5
        del primals_97
        buf1123 = buf1122[0]
        buf1125 = buf1089; del buf1089  # reuse
        buf1126 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1128 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_52.run(relu_24, buf1109, div_5, buf1123, convolution_31, unsqueeze_1958, squeeze_79, buf1125, buf1126, buf1128, 256, 8192, grid=grid(256), stream=stream0)
        buf1127 = buf1092; del buf1092  # reuse
        buf1129 = buf1127; del buf1127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53.run(buf1129, relu_24, buf1109, div_5, buf1123, convolution_31, unsqueeze_1958, buf1126, squeeze_79, buf1125, primals_95, 2097152, grid=grid(2097152), stream=stream0)
        del buf1109
        del convolution_31
        del div_5
        del primals_95
        del relu_24
        del squeeze_79
        del unsqueeze_1958
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1130 = aten.convolution_backward(buf1129, relu_23, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_94
        buf1131 = buf1130[0]
        buf1133 = buf1097; del buf1097  # reuse
        buf1134 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1135 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(relu_23, buf1131, convolution_30, unsqueeze_1970, squeeze_76, buf1133, buf1134, buf1135, 128, 8192, grid=grid(128), stream=stream0)
        buf1136 = buf1131; del buf1131  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55.run(buf1136, relu_23, convolution_30, unsqueeze_1970, buf1134, squeeze_76, buf1133, primals_92, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_30
        del primals_92
        del relu_23
        del squeeze_76
        del unsqueeze_1970
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1137 = aten.convolution_backward(buf1136, relu_22, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1136
        del primals_91
        buf1138 = buf1137[0]
        buf1140 = buf1105; del buf1105  # reuse
        buf1141 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1143 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_59.run(relu_22, buf1103, buf1138, convolution_29, unsqueeze_1982, squeeze_73, buf1140, buf1141, buf1143, 512, 8192, grid=grid(512), stream=stream0)
        buf1142 = buf1107; del buf1107  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_60.run(relu_22, buf1103, buf1138, convolution_29, unsqueeze_1982, buf1141, squeeze_73, buf1140, primals_89, buf1142, 4194304, grid=grid(4194304), stream=stream0)
        del convolution_29
        del primals_89
        del squeeze_73
        del unsqueeze_1982
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1144 = aten.convolution_backward(buf1142, sum_15, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1142
        del primals_88
        del sum_15
        buf1145 = buf1144[0]
        buf1147 = buf1111; del buf1111  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_48.run(buf1145, relu_20, buf1147, 2048, 1024, grid=grid(2048), stream=stream0)
        buf1148 = reinterpret_tensor(buf933, (8, 2, 1, 128), (256, 128, 128, 1), 0); del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_49.run(buf1147, div_4, buf1148, 2048, grid=grid(2048), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1150 = aten.convolution_backward(reinterpret_tensor(buf1148, (8, 256, 1, 1), (256, 1, 0, 0), 0), relu_21, primals_86, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_86
        buf1151 = buf1150[0]
        buf13 = buf16; del buf16  # reuse
        buf12 = buf15; del buf15  # reuse
        buf1153 = buf1118; del buf1118  # reuse
        buf1154 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1156 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_50.run(convolution_27, relu_21, buf1151, buf13, buf12, buf1153, buf1154, buf1156, 64, 8, grid=grid(64), stream=stream0)
        buf1155 = buf1151; del buf1151  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_51.run(buf1155, relu_21, convolution_27, buf12, buf1154, buf13, buf1153, primals_84, 512, grid=grid(512), stream=stream0)
        del convolution_27
        del primals_84
        del relu_21
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1158 = aten.convolution_backward(buf1155, mean_4, primals_82, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_4
        del primals_82
        buf1159 = buf1158[0]
        buf1161 = buf1126; del buf1126  # reuse
        buf1162 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1164 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_52.run(relu_20, buf1145, div_4, buf1159, convolution_26, unsqueeze_2008, squeeze_67, buf1161, buf1162, buf1164, 256, 8192, grid=grid(256), stream=stream0)
        buf1163 = buf1129; del buf1129  # reuse
        buf1165 = buf1163; del buf1163  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53.run(buf1165, relu_20, buf1145, div_4, buf1159, convolution_26, unsqueeze_2008, buf1162, squeeze_67, buf1161, primals_80, 2097152, grid=grid(2097152), stream=stream0)
        del buf1145
        del convolution_26
        del div_4
        del primals_80
        del relu_20
        del squeeze_67
        del unsqueeze_2008
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1166 = aten.convolution_backward(buf1165, relu_19, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf1165
        del primals_79
        buf1167 = buf1166[0]
        buf1169 = buf1134; del buf1134  # reuse
        buf1170 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1171 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(relu_19, buf1167, convolution_25, unsqueeze_2020, squeeze_64, buf1169, buf1170, buf1171, 128, 8192, grid=grid(128), stream=stream0)
        buf1172 = buf1167; del buf1167  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55.run(buf1172, relu_19, convolution_25, unsqueeze_2020, buf1170, squeeze_64, buf1169, primals_77, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_25
        del primals_77
        del relu_19
        del squeeze_64
        del unsqueeze_2020
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1173 = aten.convolution_backward(buf1172, relu_18, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1172
        del primals_76
        buf1174 = buf1173[0]
        buf1176 = buf1103; del buf1103  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_61.run(buf1176, relu_18, relu_22, buf1138, buf1174, 4194304, grid=grid(4194304), stream=stream0)
        del relu_18
        del relu_22
        buf1177 = buf1141; del buf1141  # reuse
        buf1178 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1184 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1179 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1185 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_62.run(buf1176, convolution_24, unsqueeze_2032, convolution_23, unsqueeze_2044, squeeze_61, squeeze_58, buf1177, buf1178, buf1184, buf1179, buf1185, 512, 8192, grid=grid(512), stream=stream0)
        buf1180 = buf1174; del buf1174  # reuse
        buf1186 = buf1138; del buf1138  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_63.run(buf1176, convolution_24, unsqueeze_2032, buf1178, squeeze_61, buf1177, primals_74, convolution_23, unsqueeze_2044, buf1184, squeeze_58, primals_71, buf1180, buf1186, 4194304, grid=grid(4194304), stream=stream0)
        del buf1176
        del convolution_23
        del convolution_24
        del primals_71
        del primals_74
        del squeeze_58
        del squeeze_61
        del unsqueeze_2032
        del unsqueeze_2044
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1181 = aten.convolution_backward(buf1180, avg_pool2d_1, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_1
        del buf1180
        del primals_73
        buf1182 = buf1181[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1187 = aten.convolution_backward(buf1186, avg_pool2d, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d
        del primals_70
        buf1188 = buf1187[0]
        buf1190 = reinterpret_tensor(buf1186, (8, 128, 64, 64), (524288, 4096, 64, 1), 0); del buf1186  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_64.run(buf1188, buf1190, 4194304, grid=grid(4194304), stream=stream0)
        del buf1188
        buf1191 = buf1147; del buf1147  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_65.run(buf1190, relu_16, buf1191, 2048, 4096, grid=grid(2048), stream=stream0)
        buf1192 = reinterpret_tensor(buf896, (8, 2, 1, 128), (256, 128, 128, 1), 0); del buf896  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_49.run(buf1191, div_3, buf1192, 2048, grid=grid(2048), stream=stream0)
        del buf1191
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1194 = aten.convolution_backward(reinterpret_tensor(buf1192, (8, 256, 1, 1), (256, 1, 0, 0), 0), relu_17, primals_68, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_68
        buf1195 = buf1194[0]
        buf10 = buf13; del buf13  # reuse
        buf9 = buf12; del buf12  # reuse
        buf1197 = buf1154; del buf1154  # reuse
        buf1198 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1200 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_50.run(convolution_21, relu_17, buf1195, buf10, buf9, buf1197, buf1198, buf1200, 64, 8, grid=grid(64), stream=stream0)
        buf1199 = buf1195; del buf1195  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_51.run(buf1199, relu_17, convolution_21, buf9, buf1198, buf10, buf1197, primals_66, 512, grid=grid(512), stream=stream0)
        del convolution_21
        del primals_66
        del relu_17
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1202 = aten.convolution_backward(buf1199, mean_3, primals_64, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_3
        del primals_64
        buf1203 = buf1202[0]
        buf1205 = buf1162; del buf1162  # reuse
        buf1206 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1208 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_16, buf1190, div_3, buf1203, convolution_20, unsqueeze_2070, squeeze_52, buf1205, buf1206, buf1208, 256, 32768, grid=grid(256), stream=stream0)
        buf1207 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        buf1209 = buf1207; del buf1207  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf1209, relu_16, buf1190, div_3, buf1203, convolution_20, unsqueeze_2070, buf1206, squeeze_52, buf1205, primals_62, 8388608, grid=grid(8388608), stream=stream0)
        del buf1190
        del convolution_20
        del div_3
        del primals_62
        del relu_16
        del squeeze_52
        del unsqueeze_2070
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1210 = aten.convolution_backward(buf1209, relu_15, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_61
        buf1211 = buf1210[0]
        buf1213 = reinterpret_tensor(buf1184, (128, 4), (1, 128), 0); del buf1184  # reuse
        buf1215 = reinterpret_tensor(buf1178, (128, 4), (1, 128), 0); del buf1178  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_15, buf1211, convolution_19, unsqueeze_2082, buf1213, buf1215, 512, 8192, grid=grid(512), stream=stream0)
        buf1214 = buf1170; del buf1170  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf1213, buf1214, 128, 4, grid=grid(128), stream=stream0)
        buf1216 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1217 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_70.run(buf1215, squeeze_49, buf1216, buf1217, 128, 4, grid=grid(128), stream=stream0)
        buf1218 = buf1211; del buf1211  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_71.run(buf1218, relu_15, convolution_19, unsqueeze_2082, buf1216, squeeze_49, buf1214, primals_59, 4194304, grid=grid(4194304), stream=stream0)
        del convolution_19
        del primals_59
        del relu_15
        del squeeze_49
        del unsqueeze_2082
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1219 = aten.convolution_backward(buf1218, relu_14, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_58
        buf1220 = buf1219[0]
        buf1222 = buf1206; del buf1206  # reuse
        buf1223 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1225 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_72.run(relu_14, buf1182, buf1220, convolution_18, unsqueeze_2094, squeeze_46, buf1222, buf1223, buf1225, 256, 32768, grid=grid(256), stream=stream0)
        buf1224 = buf1209; del buf1209  # reuse
        buf1226 = buf1224; del buf1224  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(buf1226, relu_14, buf1182, buf1220, convolution_18, unsqueeze_2094, buf1223, squeeze_46, buf1222, primals_56, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_18
        del primals_56
        del squeeze_46
        del unsqueeze_2094
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1227 = aten.convolution_backward(buf1226, sum_9, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_55
        del sum_9
        buf1228 = buf1227[0]
        buf1230 = reinterpret_tensor(buf1203, (8, 2, 64, 1, 1), (128, 64, 1, 1, 1), 0); del buf1203  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_74.run(buf1228, relu_12, buf1230, 1024, 4096, grid=grid(1024), stream=stream0)
        buf1231 = reinterpret_tensor(buf1159, (8, 2, 1, 64), (128, 64, 64, 1), 0); del buf1159  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_75.run(buf1230, div_2, buf1231, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1233 = aten.convolution_backward(reinterpret_tensor(buf1231, (8, 128, 1, 1), (128, 1, 0, 0), 0), relu_13, primals_53, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_53
        buf1234 = buf1233[0]
        buf7 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf6 = empty((1, 32, 1, 1), device='cuda', dtype=torch.float32)
        buf1236 = empty((32, ), device='cuda', dtype=torch.float32)
        buf1237 = empty((32, ), device='cuda', dtype=torch.float32)
        buf1239 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_76.run(convolution_16, relu_13, buf1234, buf7, buf6, buf1236, buf1237, buf1239, 32, 8, grid=grid(32), stream=stream0)
        buf1238 = buf1234; del buf1234  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_77.run(buf1238, relu_13, convolution_16, buf6, buf1237, buf7, buf1236, primals_51, 256, grid=grid(256), stream=stream0)
        del convolution_16
        del primals_51
        del relu_13
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1241 = aten.convolution_backward(buf1238, mean_2, primals_49, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_2
        del primals_49
        buf1242 = buf1241[0]
        buf1244 = buf1215; del buf1215  # reuse
        buf1246 = buf1213; del buf1213  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_12, buf1228, div_2, buf1242, convolution_15, unsqueeze_2120, buf1244, buf1246, 512, 8192, grid=grid(512), stream=stream0)
        buf1245 = buf1216; del buf1216  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf1244, buf1245, 128, 4, grid=grid(128), stream=stream0)
        buf1247 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1249 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_70.run(buf1246, squeeze_40, buf1247, buf1249, 128, 4, grid=grid(128), stream=stream0)
        buf1248 = buf1218; del buf1218  # reuse
        buf1250 = buf1248; del buf1248  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79.run(buf1250, relu_12, buf1228, div_2, buf1242, convolution_15, unsqueeze_2120, buf1247, squeeze_40, buf1245, primals_47, 4194304, grid=grid(4194304), stream=stream0)
        del buf1228
        del convolution_15
        del div_2
        del primals_47
        del relu_12
        del squeeze_40
        del unsqueeze_2120
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1251 = aten.convolution_backward(buf1250, relu_11, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_46
        buf1252 = buf1251[0]
        buf1254 = reinterpret_tensor(buf1223, (64, 4), (1, 64), 0); del buf1223  # reuse
        buf1256 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_11, buf1252, convolution_14, unsqueeze_2132, buf1254, buf1256, 256, 8192, grid=grid(256), stream=stream0)
        buf1255 = reinterpret_tensor(buf9, (64, ), (1, ), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1254, buf1255, 64, 4, grid=grid(64), stream=stream0)
        buf1257 = buf1198; del buf1198  # reuse
        buf1258 = reinterpret_tensor(buf10, (64, ), (1, ), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_82.run(buf1256, squeeze_37, buf1257, buf1258, 64, 4, grid=grid(64), stream=stream0)
        buf1259 = buf1252; del buf1252  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf1259, relu_11, convolution_14, unsqueeze_2132, buf1257, squeeze_37, buf1255, primals_44, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_14
        del primals_44
        del relu_11
        del squeeze_37
        del unsqueeze_2132
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1260 = aten.convolution_backward(buf1259, relu_10, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1259
        del primals_43
        buf1261 = buf1260[0]
        buf1263 = buf1220; del buf1220  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.threshold_backward]
        triton_poi_fused_add_avg_pool2d_backward_threshold_backward_84.run(buf1263, relu_10, relu_14, buf1182, buf1261, 8388608, grid=grid(8388608), stream=stream0)
        del buf1182
        del relu_10
        del relu_14
        buf1264 = reinterpret_tensor(buf1256, (256, ), (1, ), 0); del buf1256  # reuse
        buf1265 = reinterpret_tensor(buf1254, (256, ), (1, ), 0); del buf1254  # reuse
        buf1266 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_85.run(buf1263, convolution_13, unsqueeze_2144, squeeze_34, buf1264, buf1265, buf1266, 256, 32768, grid=grid(256), stream=stream0)
        buf1267 = buf1261; del buf1261  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_86.run(buf1263, convolution_13, unsqueeze_2144, buf1265, squeeze_34, buf1264, primals_41, buf1267, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_13
        del primals_41
        del squeeze_34
        del unsqueeze_2144
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1268 = aten.convolution_backward(buf1267, sum_6, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_40
        del sum_6
        buf1269 = buf1268[0]
        buf1271 = buf1230; del buf1230  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_74.run(buf1269, relu_8, buf1271, 1024, 4096, grid=grid(1024), stream=stream0)
        buf1272 = reinterpret_tensor(buf1123, (8, 2, 1, 64), (128, 64, 64, 1), 0); del buf1123  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_75.run(buf1271, div_1, buf1272, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1274 = aten.convolution_backward(reinterpret_tensor(buf1272, (8, 128, 1, 1), (128, 1, 0, 0), 0), relu_9, primals_38, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_38
        buf1275 = buf1274[0]
        buf4 = buf7; del buf7  # reuse
        buf3 = buf6; del buf6  # reuse
        buf1277 = buf1237; del buf1237  # reuse
        buf1278 = empty((32, ), device='cuda', dtype=torch.float32)
        buf1280 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_76.run(convolution_11, relu_9, buf1275, buf4, buf3, buf1277, buf1278, buf1280, 32, 8, grid=grid(32), stream=stream0)
        buf1279 = buf1275; del buf1275  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_77.run(buf1279, relu_9, convolution_11, buf3, buf1278, buf4, buf1277, primals_36, 256, grid=grid(256), stream=stream0)
        del convolution_11
        del primals_36
        del relu_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1282 = aten.convolution_backward(buf1279, mean_1, primals_34, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_34
        buf1283 = buf1282[0]
        buf1285 = reinterpret_tensor(buf1242, (128, 4), (1, 128), 0); del buf1242  # reuse
        buf1287 = buf1246; del buf1246  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_8, buf1269, div_1, buf1283, convolution_10, unsqueeze_2170, buf1285, buf1287, 512, 8192, grid=grid(512), stream=stream0)
        buf1286 = buf1247; del buf1247  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf1285, buf1286, 128, 4, grid=grid(128), stream=stream0)
        buf1288 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1290 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_70.run(buf1287, squeeze_28, buf1288, buf1290, 128, 4, grid=grid(128), stream=stream0)
        buf1289 = buf1250; del buf1250  # reuse
        buf1291 = buf1289; del buf1289  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79.run(buf1291, relu_8, buf1269, div_1, buf1283, convolution_10, unsqueeze_2170, buf1288, squeeze_28, buf1286, primals_32, 4194304, grid=grid(4194304), stream=stream0)
        del buf1269
        del convolution_10
        del div_1
        del primals_32
        del relu_8
        del squeeze_28
        del unsqueeze_2170
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1292 = aten.convolution_backward(buf1291, relu_7, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del primals_31
        buf1293 = buf1292[0]
        buf1295 = reinterpret_tensor(buf1265, (64, 4), (1, 64), 0); del buf1265  # reuse
        buf1297 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_7, buf1293, convolution_9, unsqueeze_2182, buf1295, buf1297, 256, 8192, grid=grid(256), stream=stream0)
        buf1296 = buf1257; del buf1257  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1295, buf1296, 64, 4, grid=grid(64), stream=stream0)
        buf1298 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1299 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_82.run(buf1297, squeeze_25, buf1298, buf1299, 64, 4, grid=grid(64), stream=stream0)
        buf1300 = buf1293; del buf1293  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf1300, relu_7, convolution_9, unsqueeze_2182, buf1298, squeeze_25, buf1296, primals_29, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_9
        del primals_29
        del relu_7
        del squeeze_25
        del unsqueeze_2182
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1301 = aten.convolution_backward(buf1300, relu_6, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1300
        del primals_28
        buf1302 = buf1301[0]
        buf1304 = reinterpret_tensor(buf1297, (256, ), (1, ), 0); del buf1297  # reuse
        buf1305 = reinterpret_tensor(buf1295, (256, ), (1, ), 0); del buf1295  # reuse
        buf1311 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1307 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1313 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_87.run(relu_6, buf1263, buf1302, convolution_8, unsqueeze_2194, convolution_7, unsqueeze_2206, squeeze_22, squeeze_19, buf1304, buf1305, buf1311, buf1307, buf1313, 256, 32768, grid=grid(256), stream=stream0)
        buf1306 = buf1267; del buf1267  # reuse
        buf1312 = buf1226; del buf1226  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_88.run(relu_6, buf1263, buf1302, convolution_8, unsqueeze_2194, buf1305, squeeze_22, buf1304, primals_26, convolution_7, unsqueeze_2206, buf1311, squeeze_19, primals_23, buf1306, buf1312, 8388608, grid=grid(8388608), stream=stream0)
        del buf1263
        del buf1302
        del convolution_7
        del convolution_8
        del primals_23
        del primals_26
        del relu_6
        del squeeze_19
        del squeeze_22
        del unsqueeze_2194
        del unsqueeze_2206
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1314 = aten.convolution_backward(buf1312, sum_3, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1312
        del primals_22
        del sum_3
        buf1315 = buf1314[0]
        buf1317 = buf1271; del buf1271  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_74.run(buf1315, relu_4, buf1317, 1024, 4096, grid=grid(1024), stream=stream0)
        buf1318 = reinterpret_tensor(buf1086, (8, 2, 1, 64), (128, 64, 64, 1), 0); del buf1086  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_poi_fused__softmax_backward_data_75.run(buf1317, div, buf1318, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1320 = aten.convolution_backward(reinterpret_tensor(buf1318, (8, 128, 1, 1), (128, 1, 0, 0), 0), relu_5, primals_20, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_20
        buf1321 = buf1320[0]
        buf1 = buf4; del buf4  # reuse
        buf0 = buf3; del buf3  # reuse
        buf1323 = buf1278; del buf1278  # reuse
        buf1324 = empty((32, ), device='cuda', dtype=torch.float32)
        buf1326 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_76.run(convolution_5, relu_5, buf1321, buf1, buf0, buf1323, buf1324, buf1326, 32, 8, grid=grid(32), stream=stream0)
        buf100 = empty((1000, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view_198, out=buf100)
        del view_198
        buf101 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_89.run(tangents_1, buf101, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf108 = buf106[1]
        del buf106
        buf111 = reinterpret_tensor(buf1317, (1024, ), (1, ), 0); del buf1317  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_90.run(buf110, buf111, 1024, 8, grid=grid(1024), stream=stream0)
        del buf110
        buf114 = buf112[1]
        del buf112
        buf119 = buf1311; del buf1311  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf117, buf119, 256, 8, grid=grid(256), stream=stream0)
        del buf117
        buf122 = buf120[1]
        del buf120
        buf130 = buf128[1]
        del buf128
        buf137 = buf135[1]
        del buf135
        buf145 = buf143[1]
        del buf143
        buf148 = buf1030; del buf1030  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_90.run(buf147, buf148, 1024, 8, grid=grid(1024), stream=stream0)
        del buf147
        buf151 = buf149[1]
        del buf149
        buf156 = buf1305; del buf1305  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf154, buf156, 256, 8, grid=grid(256), stream=stream0)
        del buf154
        buf159 = buf157[1]
        del buf157
        buf167 = buf165[1]
        del buf165
        buf174 = buf172[1]
        del buf172
        buf182 = buf180[1]
        del buf180
        buf188 = buf186[1]
        del buf186
        buf192 = buf1024; del buf1024  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_90.run(buf191, buf192, 1024, 8, grid=grid(1024), stream=stream0)
        del buf191
        buf195 = buf193[1]
        del buf193
        buf200 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf198, buf200, 256, 8, grid=grid(256), stream=stream0)
        del buf198
        buf203 = buf201[1]
        del buf201
        buf211 = buf209[1]
        del buf209
        buf218 = buf216[1]
        del buf216
        buf226 = buf224[1]
        del buf224
        buf229 = reinterpret_tensor(buf1283, (512, ), (1, ), 0); del buf1283  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf228, buf229, 512, 8, grid=grid(512), stream=stream0)
        del buf228
        buf232 = buf230[1]
        del buf230
        buf237 = buf1288; del buf1288  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf235, buf237, 128, 8, grid=grid(128), stream=stream0)
        del buf235
        buf240 = buf238[1]
        del buf238
        buf248 = buf246[1]
        del buf246
        buf255 = buf253[1]
        del buf253
        buf263 = buf261[1]
        del buf261
        buf266 = reinterpret_tensor(buf1287, (512, ), (1, ), 0); del buf1287  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf265, buf266, 512, 8, grid=grid(512), stream=stream0)
        del buf265
        buf269 = buf267[1]
        del buf267
        buf274 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf272, buf274, 128, 8, grid=grid(128), stream=stream0)
        del buf272
        buf277 = buf275[1]
        del buf275
        buf285 = buf283[1]
        del buf283
        buf292 = buf290[1]
        del buf290
        buf299 = buf297[1]
        del buf297
        buf302 = reinterpret_tensor(buf1285, (512, ), (1, ), 0); del buf1285  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf301, buf302, 512, 8, grid=grid(512), stream=stream0)
        del buf301
        buf305 = buf303[1]
        del buf303
        buf310 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf308, buf310, 128, 8, grid=grid(128), stream=stream0)
        del buf308
        buf313 = buf311[1]
        del buf311
        buf321 = buf319[1]
        del buf319
        buf328 = buf326[1]
        del buf326
        buf336 = buf334[1]
        del buf334
        buf339 = reinterpret_tensor(buf1244, (512, ), (1, ), 0); del buf1244  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf338, buf339, 512, 8, grid=grid(512), stream=stream0)
        del buf338
        buf342 = buf340[1]
        del buf340
        buf347 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf345, buf347, 128, 8, grid=grid(128), stream=stream0)
        del buf345
        buf350 = buf348[1]
        del buf348
        buf358 = buf356[1]
        del buf356
        buf365 = buf363[1]
        del buf363
        buf372 = buf370[1]
        del buf370
        buf375 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf374, buf375, 512, 8, grid=grid(512), stream=stream0)
        del buf374
        buf378 = buf376[1]
        del buf376
        buf383 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf381, buf383, 128, 8, grid=grid(128), stream=stream0)
        del buf381
        buf386 = buf384[1]
        del buf384
        buf394 = buf392[1]
        del buf392
        buf401 = buf399[1]
        del buf399
        buf409 = buf407[1]
        del buf407
        buf412 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf411, buf412, 512, 8, grid=grid(512), stream=stream0)
        del buf411
        buf415 = buf413[1]
        del buf413
        buf420 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf418, buf420, 128, 8, grid=grid(128), stream=stream0)
        del buf418
        buf423 = buf421[1]
        del buf421
        buf431 = buf429[1]
        del buf429
        buf438 = buf436[1]
        del buf436
        buf445 = buf443[1]
        del buf443
        buf448 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf447, buf448, 512, 8, grid=grid(512), stream=stream0)
        del buf447
        buf451 = buf449[1]
        del buf449
        buf456 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf454, buf456, 128, 8, grid=grid(128), stream=stream0)
        del buf454
        buf459 = buf457[1]
        del buf457
        buf467 = buf465[1]
        del buf465
        buf474 = buf472[1]
        del buf472
        buf482 = buf480[1]
        del buf480
        buf485 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf484, buf485, 512, 8, grid=grid(512), stream=stream0)
        del buf484
        buf488 = buf486[1]
        del buf486
        buf493 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf491, buf493, 128, 8, grid=grid(128), stream=stream0)
        del buf491
        buf496 = buf494[1]
        del buf494
        buf504 = buf502[1]
        del buf502
        buf511 = buf509[1]
        del buf509
        buf518 = buf516[1]
        del buf516
        buf521 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf520, buf521, 512, 8, grid=grid(512), stream=stream0)
        del buf520
        buf524 = buf522[1]
        del buf522
        buf529 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf527, buf529, 128, 8, grid=grid(128), stream=stream0)
        del buf527
        buf532 = buf530[1]
        del buf530
        buf540 = buf538[1]
        del buf538
        buf547 = buf545[1]
        del buf545
        buf555 = buf553[1]
        del buf553
        buf558 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf557, buf558, 512, 8, grid=grid(512), stream=stream0)
        del buf557
        buf561 = buf559[1]
        del buf559
        buf566 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf564, buf566, 128, 8, grid=grid(128), stream=stream0)
        del buf564
        buf569 = buf567[1]
        del buf567
        buf577 = buf575[1]
        del buf575
        buf584 = buf582[1]
        del buf582
        buf591 = buf589[1]
        del buf589
        buf594 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf593, buf594, 512, 8, grid=grid(512), stream=stream0)
        del buf593
        buf597 = buf595[1]
        del buf595
        buf602 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf600, buf602, 128, 8, grid=grid(128), stream=stream0)
        del buf600
        buf605 = buf603[1]
        del buf603
        buf613 = buf611[1]
        del buf611
        buf620 = buf618[1]
        del buf618
        buf628 = buf626[1]
        del buf626
        buf631 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf630, buf631, 512, 8, grid=grid(512), stream=stream0)
        del buf630
        buf634 = buf632[1]
        del buf632
        buf639 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf637, buf639, 128, 8, grid=grid(128), stream=stream0)
        del buf637
        buf642 = buf640[1]
        del buf640
        buf650 = buf648[1]
        del buf648
        buf657 = buf655[1]
        del buf655
        buf664 = buf662[1]
        del buf662
        buf667 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf666, buf667, 512, 8, grid=grid(512), stream=stream0)
        del buf666
        buf670 = buf668[1]
        del buf668
        buf675 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf673, buf675, 128, 8, grid=grid(128), stream=stream0)
        del buf673
        buf678 = buf676[1]
        del buf676
        buf686 = buf684[1]
        del buf684
        buf693 = buf691[1]
        del buf691
        buf701 = buf699[1]
        del buf699
        buf704 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf703, buf704, 512, 8, grid=grid(512), stream=stream0)
        del buf703
        buf707 = buf705[1]
        del buf705
        buf712 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf710, buf712, 128, 8, grid=grid(128), stream=stream0)
        del buf710
        buf715 = buf713[1]
        del buf713
        buf723 = buf721[1]
        del buf721
        buf730 = buf728[1]
        del buf728
        buf737 = buf735[1]
        del buf735
        buf740 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf739, buf740, 512, 8, grid=grid(512), stream=stream0)
        del buf739
        buf743 = buf741[1]
        del buf741
        buf748 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf746, buf748, 128, 8, grid=grid(128), stream=stream0)
        del buf746
        buf751 = buf749[1]
        del buf749
        buf759 = buf757[1]
        del buf757
        buf766 = buf764[1]
        del buf764
        buf774 = buf772[1]
        del buf772
        buf777 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf776, buf777, 512, 8, grid=grid(512), stream=stream0)
        del buf776
        buf780 = buf778[1]
        del buf778
        buf785 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf783, buf785, 128, 8, grid=grid(128), stream=stream0)
        del buf783
        buf788 = buf786[1]
        del buf786
        buf796 = buf794[1]
        del buf794
        buf803 = buf801[1]
        del buf801
        buf810 = buf808[1]
        del buf808
        buf813 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf812, buf813, 512, 8, grid=grid(512), stream=stream0)
        del buf812
        buf816 = buf814[1]
        del buf814
        buf821 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf819, buf821, 128, 8, grid=grid(128), stream=stream0)
        del buf819
        buf824 = buf822[1]
        del buf822
        buf832 = buf830[1]
        del buf830
        buf839 = buf837[1]
        del buf837
        buf847 = buf845[1]
        del buf845
        buf850 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf849, buf850, 512, 8, grid=grid(512), stream=stream0)
        del buf849
        buf853 = buf851[1]
        del buf851
        buf858 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf856, buf858, 128, 8, grid=grid(128), stream=stream0)
        del buf856
        buf861 = buf859[1]
        del buf859
        buf869 = buf867[1]
        del buf867
        buf876 = buf874[1]
        del buf874
        buf883 = buf881[1]
        del buf881
        buf886 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf885, buf886, 512, 8, grid=grid(512), stream=stream0)
        del buf885
        buf889 = buf887[1]
        del buf887
        buf894 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf892, buf894, 128, 8, grid=grid(128), stream=stream0)
        del buf892
        buf897 = buf895[1]
        del buf895
        buf905 = buf903[1]
        del buf903
        buf912 = buf910[1]
        del buf910
        buf920 = buf918[1]
        del buf918
        buf923 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf922, buf923, 512, 8, grid=grid(512), stream=stream0)
        del buf922
        buf926 = buf924[1]
        del buf924
        buf931 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf929, buf931, 128, 8, grid=grid(128), stream=stream0)
        del buf929
        buf934 = buf932[1]
        del buf932
        buf942 = buf940[1]
        del buf940
        buf949 = buf947[1]
        del buf947
        buf956 = buf954[1]
        del buf954
        buf959 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf958, buf959, 512, 8, grid=grid(512), stream=stream0)
        del buf958
        buf962 = buf960[1]
        del buf960
        buf967 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf965, buf967, 128, 8, grid=grid(128), stream=stream0)
        del buf965
        buf970 = buf968[1]
        del buf968
        buf978 = buf976[1]
        del buf976
        buf985 = buf983[1]
        del buf983
        buf993 = buf991[1]
        del buf991
        buf996 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf995, buf996, 512, 8, grid=grid(512), stream=stream0)
        del buf995
        buf999 = buf997[1]
        del buf997
        buf1004 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf1002, buf1004, 128, 8, grid=grid(128), stream=stream0)
        del buf1002
        buf1007 = buf1005[1]
        del buf1005
        buf1015 = buf1013[1]
        del buf1013
        buf1022 = buf1020[1]
        del buf1020
        buf1029 = buf1027[1]
        del buf1027
        buf1035 = buf1033[1]
        del buf1033
        buf1039 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf1038, buf1039, 512, 8, grid=grid(512), stream=stream0)
        del buf1038
        buf1042 = buf1040[1]
        del buf1040
        buf1047 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf1045, buf1047, 128, 8, grid=grid(128), stream=stream0)
        del buf1045
        buf1050 = buf1048[1]
        del buf1048
        buf1058 = buf1056[1]
        del buf1056
        buf1065 = buf1063[1]
        del buf1063
        buf1073 = buf1071[1]
        del buf1071
        buf1076 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf1075, buf1076, 256, 8, grid=grid(256), stream=stream0)
        del buf1075
        buf1079 = buf1077[1]
        del buf1077
        buf1084 = buf1298; del buf1298  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_94.run(buf1082, buf1084, 64, 8, grid=grid(64), stream=stream0)
        del buf1082
        buf1087 = buf1085[1]
        del buf1085
        buf1095 = buf1093[1]
        del buf1093
        buf1102 = buf1100[1]
        del buf1100
        buf1110 = buf1108[1]
        del buf1108
        buf1113 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf1112, buf1113, 256, 8, grid=grid(256), stream=stream0)
        del buf1112
        buf1116 = buf1114[1]
        del buf1114
        buf1121 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_94.run(buf1119, buf1121, 64, 8, grid=grid(64), stream=stream0)
        del buf1119
        buf1124 = buf1122[1]
        del buf1122
        buf1132 = buf1130[1]
        del buf1130
        buf1139 = buf1137[1]
        del buf1137
        buf1146 = buf1144[1]
        del buf1144
        buf1149 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf1148, buf1149, 256, 8, grid=grid(256), stream=stream0)
        del buf1148
        buf1152 = buf1150[1]
        del buf1150
        buf1157 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_94.run(buf1155, buf1157, 64, 8, grid=grid(64), stream=stream0)
        buf1160 = buf1158[1]
        del buf1158
        buf1168 = buf1166[1]
        del buf1166
        buf1175 = buf1173[1]
        del buf1173
        buf1183 = buf1181[1]
        del buf1181
        buf1189 = buf1187[1]
        del buf1187
        buf1193 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf1192, buf1193, 256, 8, grid=grid(256), stream=stream0)
        del buf1192
        buf1196 = buf1194[1]
        del buf1194
        buf1201 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_94.run(buf1199, buf1201, 64, 8, grid=grid(64), stream=stream0)
        buf1204 = buf1202[1]
        del buf1202
        buf1212 = buf1210[1]
        del buf1210
        buf1221 = buf1219[1]
        del buf1219
        buf1229 = buf1227[1]
        del buf1227
        buf1232 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf1231, buf1232, 128, 8, grid=grid(128), stream=stream0)
        del buf1231
        buf1235 = buf1233[1]
        del buf1233
        buf1240 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_95.run(buf1238, buf1240, 32, 8, grid=grid(32), stream=stream0)
        del buf1238
        buf1243 = buf1241[1]
        del buf1241
        buf1253 = buf1251[1]
        del buf1251
        buf1262 = buf1260[1]
        del buf1260
        buf1270 = buf1268[1]
        del buf1268
        buf1273 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf1272, buf1273, 128, 8, grid=grid(128), stream=stream0)
        buf1276 = buf1274[1]
        del buf1274
        buf1281 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_95.run(buf1279, buf1281, 32, 8, grid=grid(32), stream=stream0)
        buf1284 = buf1282[1]
        del buf1282
        buf1294 = buf1292[1]
        del buf1292
        buf1303 = buf1301[1]
        del buf1301
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1308 = aten.convolution_backward(buf1306, getitem_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1306
        del primals_25
        buf1309 = buf1308[0]
        buf1310 = buf1308[1]
        del buf1308
        buf1316 = buf1314[1]
        del buf1314
        buf1319 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf1318, buf1319, 128, 8, grid=grid(128), stream=stream0)
        buf1322 = buf1320[1]
        del buf1320
        buf1325 = buf1321; del buf1321  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_77.run(buf1325, relu_5, convolution_5, buf0, buf1324, buf1, buf1323, primals_18, 256, grid=grid(256), stream=stream0)
        del buf0
        del buf1
        del convolution_5
        del primals_18
        del relu_5
        buf1327 = buf1324; del buf1324  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_95.run(buf1325, buf1327, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1328 = aten.convolution_backward(buf1325, mean, primals_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_16
        buf1329 = buf1328[0]
        buf1330 = buf1328[1]
        del buf1328
        buf1331 = reinterpret_tensor(buf1199, (128, 4), (1, 128), 0); del buf1199  # reuse
        buf1333 = reinterpret_tensor(buf1155, (128, 4), (1, 128), 0); del buf1155  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_4, buf1315, div, buf1329, convolution_4, unsqueeze_2232, buf1331, buf1333, 512, 8192, grid=grid(512), stream=stream0)
        buf1332 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf1331, buf1332, 128, 4, grid=grid(128), stream=stream0)
        del buf1331
        buf1334 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1336 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_70.run(buf1333, squeeze_13, buf1334, buf1336, 128, 4, grid=grid(128), stream=stream0)
        buf1335 = buf1291; del buf1291  # reuse
        buf1337 = buf1335; del buf1335  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79.run(buf1337, relu_4, buf1315, div, buf1329, convolution_4, unsqueeze_2232, buf1334, squeeze_13, buf1332, primals_14, 4194304, grid=grid(4194304), stream=stream0)
        del buf1315
        del convolution_4
        del div
        del primals_14
        del relu_4
        del squeeze_13
        del unsqueeze_2232
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1338 = aten.convolution_backward(buf1337, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
        del buf1337
        del primals_13
        buf1339 = buf1338[0]
        buf1340 = buf1338[1]
        del buf1338
        buf1341 = reinterpret_tensor(buf1325, (64, 4), (1, 64), 0); del buf1325  # reuse
        buf1343 = reinterpret_tensor(buf1279, (64, 4), (1, 64), 0); del buf1279  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_3, buf1339, convolution_3, unsqueeze_2244, buf1341, buf1343, 256, 8192, grid=grid(256), stream=stream0)
        buf1342 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1341, buf1342, 64, 4, grid=grid(64), stream=stream0)
        del buf1341
        buf1344 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1345 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_82.run(buf1343, squeeze_10, buf1344, buf1345, 64, 4, grid=grid(64), stream=stream0)
        del buf1343
        buf1346 = buf1339; del buf1339  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(buf1346, relu_3, convolution_3, unsqueeze_2244, buf1344, squeeze_10, buf1342, primals_11, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_3
        del primals_11
        del relu_3
        del squeeze_10
        del unsqueeze_2244
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1347 = aten.convolution_backward(buf1346, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1346
        del getitem_6
        del primals_10
        buf1348 = buf1347[0]
        buf1349 = buf1347[1]
        del buf1347
        buf1350 = buf1309; del buf1309  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_96.run(buf1350, buf1348, 4194304, grid=grid(4194304), stream=stream0)
        del buf1348
        buf1351 = empty((8, 128, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        triton_poi_fused_add_max_pool2d_with_indices_backward_97.run(getitem_7, buf1350, buf1351, 16777216, grid=grid(16777216), stream=stream0)
        del buf1350
        del getitem_7
        buf1352 = reinterpret_tensor(buf1329, (128, 4), (1, 128), 0); del buf1329  # reuse
        buf1354 = buf1333; del buf1333  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_98.run(relu_2, buf1351, convolution_2, unsqueeze_2256, buf1352, buf1354, 512, 32768, grid=grid(512), stream=stream0)
        buf1353 = buf1334; del buf1334  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf1352, buf1353, 128, 4, grid=grid(128), stream=stream0)
        del buf1352
        buf1355 = empty((128, ), device='cuda', dtype=torch.float32)
        buf1356 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_70.run(buf1354, squeeze_7, buf1355, buf1356, 128, 4, grid=grid(128), stream=stream0)
        del buf1354
        buf1357 = buf1351; del buf1351  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99.run(buf1357, relu_2, convolution_2, unsqueeze_2256, buf1355, squeeze_7, buf1353, primals_8, 16777216, grid=grid(16777216), stream=stream0)
        del buf1355
        del convolution_2
        del primals_8
        del relu_2
        del squeeze_7
        del unsqueeze_2256
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1358 = aten.convolution_backward(buf1357, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1357
        del primals_7
        buf1359 = buf1358[0]
        buf1360 = buf1358[1]
        del buf1358
        buf1361 = reinterpret_tensor(buf1318, (64, 16), (16, 1), 0); del buf1318  # reuse
        buf1363 = reinterpret_tensor(buf1272, (64, 16), (16, 1), 0); del buf1272  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(relu_1, buf1359, convolution_1, unsqueeze_2268, buf1361, buf1363, 1024, 8192, grid=grid(1024), stream=stream0)
        buf1362 = buf1344; del buf1344  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_101.run(buf1361, buf1362, 64, 16, grid=grid(64), stream=stream0)
        buf1364 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1365 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_102.run(buf1363, squeeze_4, buf1364, buf1365, 64, 16, grid=grid(64), stream=stream0)
        buf1366 = buf1359; del buf1359  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103.run(buf1366, relu_1, convolution_1, unsqueeze_2268, buf1364, squeeze_4, buf1362, primals_5, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_1
        del primals_5
        del relu_1
        del squeeze_4
        del unsqueeze_2268
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1367 = aten.convolution_backward(buf1366, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1366
        del primals_4
        buf1368 = buf1367[0]
        buf1369 = buf1367[1]
        del buf1367
        buf1370 = buf1363; del buf1363  # reuse
        buf1372 = buf1361; del buf1361  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_100.run(relu, buf1368, convolution, unsqueeze_2280, buf1370, buf1372, 1024, 8192, grid=grid(1024), stream=stream0)
        buf1371 = buf1364; del buf1364  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_101.run(buf1370, buf1371, 64, 16, grid=grid(64), stream=stream0)
        del buf1370
        buf1373 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1374 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_102.run(buf1372, squeeze_1, buf1373, buf1374, 64, 16, grid=grid(64), stream=stream0)
        del buf1372
        buf1375 = buf1368; del buf1368  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103.run(buf1375, relu, convolution, unsqueeze_2280, buf1373, squeeze_1, buf1371, primals_2, 8388608, grid=grid(8388608), stream=stream0)
        del buf1373
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_2280
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1376 = aten.convolution_backward(buf1375, primals_936, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1375
        del primals_1
        del primals_936
        buf1377 = buf1376[1]
        return (buf1377, buf1374, buf1371, buf1369, buf1365, buf1362, buf1360, buf1356, buf1353, buf1349, buf1345, buf1342, buf1340, buf1336, buf1332, buf1330, buf1327, buf1326, buf1323, buf1322, buf1319, buf1316, buf1313, buf1304, buf1310, buf1307, buf1304, buf1303, buf1299, buf1296, buf1294, buf1290, buf1286, buf1284, buf1281, buf1280, buf1277, buf1276, buf1273, buf1270, buf1266, buf1264, buf1262, buf1258, buf1255, buf1253, buf1249, buf1245, buf1243, buf1240, buf1239, buf1236, buf1235, buf1232, buf1229, buf1225, buf1222, buf1221, buf1217, buf1214, buf1212, buf1208, buf1205, buf1204, buf1201, buf1200, buf1197, buf1196, buf1193, buf1189, buf1185, buf1177, buf1183, buf1179, buf1177, buf1175, buf1171, buf1169, buf1168, buf1164, buf1161, buf1160, buf1157, buf1156, buf1153, buf1152, buf1149, buf1146, buf1143, buf1140, buf1139, buf1135, buf1133, buf1132, buf1128, buf1125, buf1124, buf1121, buf1120, buf1117, buf1116, buf1113, buf1110, buf1106, buf1104, buf1102, buf1098, buf1096, buf1095, buf1091, buf1088, buf1087, buf1084, buf1083, buf1080, buf1079, buf1076, buf1073, buf1069, buf1066, buf1065, buf1061, buf1059, buf1058, buf1054, buf1051, buf1050, buf1047, buf1046, buf1043, buf1042, buf1039, buf1035, buf1032, buf1023, buf1029, buf1026, buf1023, buf1022, buf1018, buf1016, buf1015, buf1011, buf1008, buf1007, buf1004, buf1003, buf1000, buf999, buf996, buf993, buf989, buf987, buf985, buf981, buf979, buf978, buf974, buf971, buf970, buf967, buf966, buf963, buf962, buf959, buf956, buf953, buf950, buf949, buf945, buf943, buf942, buf938, buf935, buf934, buf931, buf930, buf927, buf926, buf923, buf920, buf916, buf914, buf912, buf908, buf906, buf905, buf901, buf898, buf897, buf894, buf893, buf890, buf889, buf886, buf883, buf880, buf877, buf876, buf872, buf870, buf869, buf865, buf862, buf861, buf858, buf857, buf854, buf853, buf850, buf847, buf843, buf841, buf839, buf835, buf833, buf832, buf828, buf825, buf824, buf821, buf820, buf817, buf816, buf813, buf810, buf807, buf804, buf803, buf799, buf797, buf796, buf792, buf789, buf788, buf785, buf784, buf781, buf780, buf777, buf774, buf770, buf768, buf766, buf762, buf760, buf759, buf755, buf752, buf751, buf748, buf747, buf744, buf743, buf740, buf737, buf734, buf731, buf730, buf726, buf724, buf723, buf719, buf716, buf715, buf712, buf711, buf708, buf707, buf704, buf701, buf697, buf695, buf693, buf689, buf687, buf686, buf682, buf679, buf678, buf675, buf674, buf671, buf670, buf667, buf664, buf661, buf658, buf657, buf653, buf651, buf650, buf646, buf643, buf642, buf639, buf638, buf635, buf634, buf631, buf628, buf624, buf622, buf620, buf616, buf614, buf613, buf609, buf606, buf605, buf602, buf601, buf598, buf597, buf594, buf591, buf588, buf585, buf584, buf580, buf578, buf577, buf573, buf570, buf569, buf566, buf565, buf562, buf561, buf558, buf555, buf551, buf549, buf547, buf543, buf541, buf540, buf536, buf533, buf532, buf529, buf528, buf525, buf524, buf521, buf518, buf515, buf512, buf511, buf507, buf505, buf504, buf500, buf497, buf496, buf493, buf492, buf489, buf488, buf485, buf482, buf478, buf476, buf474, buf470, buf468, buf467, buf463, buf460, buf459, buf456, buf455, buf452, buf451, buf448, buf445, buf442, buf439, buf438, buf434, buf432, buf431, buf427, buf424, buf423, buf420, buf419, buf416, buf415, buf412, buf409, buf405, buf403, buf401, buf397, buf395, buf394, buf390, buf387, buf386, buf383, buf382, buf379, buf378, buf375, buf372, buf369, buf366, buf365, buf361, buf359, buf358, buf354, buf351, buf350, buf347, buf346, buf343, buf342, buf339, buf336, buf332, buf330, buf328, buf324, buf322, buf321, buf317, buf314, buf313, buf310, buf309, buf306, buf305, buf302, buf299, buf296, buf293, buf292, buf288, buf286, buf285, buf281, buf278, buf277, buf274, buf273, buf270, buf269, buf266, buf263, buf259, buf257, buf255, buf251, buf249, buf248, buf244, buf241, buf240, buf237, buf236, buf233, buf232, buf229, buf226, buf222, buf219, buf218, buf214, buf212, buf211, buf207, buf204, buf203, buf200, buf199, buf196, buf195, buf192, buf188, buf184, buf176, buf182, buf178, buf176, buf174, buf170, buf168, buf167, buf163, buf160, buf159, buf156, buf155, buf152, buf151, buf148, buf145, buf141, buf138, buf137, buf133, buf131, buf130, buf126, buf123, buf122, buf119, buf118, buf115, buf114, buf111, buf108, buf104, buf102, reinterpret_tensor(buf100, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf101, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.int64)
    convolution_3 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 2, 1, 64), (128, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    sum_3 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 2, 1, 64), (128, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    sum_6 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 2, 1, 64), (128, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    sum_9 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 2, 1, 128), (256, 128, 128, 1), device='cuda:0', dtype=torch.float32)
    sum_12 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 2, 1, 128), (256, 128, 128, 1), device='cuda:0', dtype=torch.float32)
    sum_15 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 2, 1, 128), (256, 128, 128, 1), device='cuda:0', dtype=torch.float32)
    sum_18 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 2, 1, 128), (256, 128, 128, 1), device='cuda:0', dtype=torch.float32)
    sum_21 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_24 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_3 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_27 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_30 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_33 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_48 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_49 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_36 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_52 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_12 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_53 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_39 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_54 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_56 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_13 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_57 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_42 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_58 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_59 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_60 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_14 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_61 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_45 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_62 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_63 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_64 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_15 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_48 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_66 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_67 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_68 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_16 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_69 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_51 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_90 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_71 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_72 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_17 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_73 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_54 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_74 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_96 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_97 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_76 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_18 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_98 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_77 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_57 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_100 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_78 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_101 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_79 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_102 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_80 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_19 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_103 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_81 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_60 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_105 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_256 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_82 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_106 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_83 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_107 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_262 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_84 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_20 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_108 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_85 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_63 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_110 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_268 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_86 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_111 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_271 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_87 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_112 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_88 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_21 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_113 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_89 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_66 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_115 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_90 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_116 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_91 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_117 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_92 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_22 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_118 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_93 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_69 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_120 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_292 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_94 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_121 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_295 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_95 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_122 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_96 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_23 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_123 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_97 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_72 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_125 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_304 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_98 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_126 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_307 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_99 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_127 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_100 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_24 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_128 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_101 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_75 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_130 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_316 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_102 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_131 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_319 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_103 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_132 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_104 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_25 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_133 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_105 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_78 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_135 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_328 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_106 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_136 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_107 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_137 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_334 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_108 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_26 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_138 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_109 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_81 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_140 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_340 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_110 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_141 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_343 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_111 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_142 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_346 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_112 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_27 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_143 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_113 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_84 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_145 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_352 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_114 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_146 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_355 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_115 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_147 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_358 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_116 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_28 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_148 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_117 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_87 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_150 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_364 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_118 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_151 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_367 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_119 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_152 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_370 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_120 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_29 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_153 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_121 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((8, 2, 1, 256), (512, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    sum_90 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_155 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_376 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_122 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_156 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_379 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_123 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_157 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_382 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_124 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mean_30 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_158 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_125 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 2, 1, 512), (1024, 512, 512, 1), device='cuda:0', dtype=torch.float32)
    sum_93 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    avg_pool2d_4 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_160 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_388 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_5 = rand_strided((8, 1024, 8, 8), (65536, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_161 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_391 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_126 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_162 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_394 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_127 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_163 = rand_strided((8, 1024, 8, 8), (65536, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_397 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_128 = rand_strided((8, 1024, 8, 8), (65536, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    mean_31 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_164 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_129 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((8, 2, 1, 512), (1024, 512, 512, 1), device='cuda:0', dtype=torch.float32)
    sum_96 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_166 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_403 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_130 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_167 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_406 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_131 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_168 = rand_strided((8, 1024, 8, 8), (65536, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_409 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_132 = rand_strided((8, 1024, 8, 8), (65536, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    mean_32 = rand_strided((8, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    convolution_169 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    relu_133 = rand_strided((8, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 2, 1, 512), (1024, 512, 512, 1), device='cuda:0', dtype=torch.float32)
    sum_99 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_171 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_415 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_34 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_584 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_596 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_608 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_696 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_708 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_720 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_770 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_796 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_808 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_820 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_870 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_896 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_908 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_920 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_946 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_958 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_970 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_996 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1008 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1020 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1046 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1058 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1070 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1096 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1108 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1120 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1146 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1158 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1170 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1196 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1208 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1220 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1246 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1258 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1270 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1296 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1308 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1320 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1346 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1358 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1370 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1396 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1408 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1420 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1446 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1458 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1470 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1496 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1508 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1520 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1546 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1558 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1570 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1596 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1608 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1620 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1646 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1658 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1670 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1696 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1708 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1720 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1746 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1758 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1770 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1796 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1808 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1820 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1832 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1858 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1870 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1882 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1908 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1920 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1932 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1958 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1970 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1982 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2008 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2020 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2032 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2044 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2070 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2082 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2094 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2120 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2132 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2144 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2170 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2182 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2194 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2206 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2232 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2244 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2256 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2268 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2280 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_51, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_66, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_99, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_114, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_129, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_147, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_162, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_177, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_192, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_207, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_222, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_237, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_252, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_267, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_282, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_297, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_312, primals_314, primals_316, primals_317, primals_319, primals_320, primals_322, primals_323, primals_325, primals_327, primals_329, primals_331, primals_332, primals_334, primals_335, primals_337, primals_338, primals_340, primals_342, primals_344, primals_346, primals_347, primals_349, primals_350, primals_352, primals_353, primals_355, primals_357, primals_359, primals_361, primals_362, primals_364, primals_365, primals_367, primals_368, primals_370, primals_372, primals_374, primals_376, primals_377, primals_379, primals_380, primals_382, primals_383, primals_385, primals_387, primals_389, primals_391, primals_392, primals_394, primals_395, primals_397, primals_398, primals_400, primals_402, primals_404, primals_406, primals_407, primals_409, primals_410, primals_412, primals_413, primals_415, primals_417, primals_419, primals_421, primals_422, primals_424, primals_425, primals_427, primals_428, primals_430, primals_432, primals_434, primals_436, primals_437, primals_439, primals_440, primals_442, primals_443, primals_445, primals_447, primals_449, primals_451, primals_452, primals_454, primals_455, primals_457, primals_458, primals_460, primals_462, primals_464, primals_466, primals_467, primals_469, primals_470, primals_472, primals_473, primals_475, primals_477, primals_479, primals_481, primals_482, primals_484, primals_485, primals_487, primals_488, primals_490, primals_491, primals_493, primals_495, primals_497, primals_499, primals_500, primals_502, primals_503, primals_505, primals_506, primals_508, primals_510, primals_512, primals_514, primals_515, primals_936, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, mean, convolution_5, relu_5, div, sum_3, convolution_7, squeeze_19, convolution_8, squeeze_22, relu_6, convolution_9, squeeze_25, relu_7, convolution_10, squeeze_28, relu_8, mean_1, convolution_11, relu_9, div_1, sum_6, convolution_13, squeeze_34, relu_10, convolution_14, squeeze_37, relu_11, convolution_15, squeeze_40, relu_12, mean_2, convolution_16, relu_13, div_2, sum_9, convolution_18, squeeze_46, relu_14, convolution_19, squeeze_49, relu_15, convolution_20, squeeze_52, relu_16, mean_3, convolution_21, relu_17, div_3, sum_12, avg_pool2d, convolution_23, squeeze_58, avg_pool2d_1, convolution_24, squeeze_61, relu_18, convolution_25, squeeze_64, relu_19, convolution_26, squeeze_67, relu_20, mean_4, convolution_27, relu_21, div_4, sum_15, convolution_29, squeeze_73, relu_22, convolution_30, squeeze_76, relu_23, convolution_31, squeeze_79, relu_24, mean_5, convolution_32, relu_25, div_5, sum_18, convolution_34, squeeze_85, relu_26, convolution_35, squeeze_88, relu_27, convolution_36, squeeze_91, relu_28, mean_6, convolution_37, relu_29, div_6, sum_21, convolution_39, squeeze_97, relu_30, convolution_40, squeeze_100, relu_31, convolution_41, squeeze_103, relu_32, mean_7, convolution_42, relu_33, div_7, sum_24, avg_pool2d_2, convolution_44, squeeze_109, avg_pool2d_3, convolution_45, squeeze_112, relu_34, convolution_46, squeeze_115, relu_35, convolution_47, squeeze_118, relu_36, mean_8, convolution_48, relu_37, div_8, sum_27, convolution_50, squeeze_124, relu_38, convolution_51, squeeze_127, relu_39, convolution_52, squeeze_130, relu_40, mean_9, convolution_53, relu_41, div_9, sum_30, convolution_55, squeeze_136, relu_42, convolution_56, squeeze_139, relu_43, convolution_57, squeeze_142, relu_44, mean_10, convolution_58, relu_45, div_10, sum_33, convolution_60, squeeze_148, relu_46, convolution_61, squeeze_151, relu_47, convolution_62, squeeze_154, relu_48, mean_11, convolution_63, relu_49, div_11, sum_36, convolution_65, squeeze_160, relu_50, convolution_66, squeeze_163, relu_51, convolution_67, squeeze_166, relu_52, mean_12, convolution_68, relu_53, div_12, sum_39, convolution_70, squeeze_172, relu_54, convolution_71, squeeze_175, relu_55, convolution_72, squeeze_178, relu_56, mean_13, convolution_73, relu_57, div_13, sum_42, convolution_75, squeeze_184, relu_58, convolution_76, squeeze_187, relu_59, convolution_77, squeeze_190, relu_60, mean_14, convolution_78, relu_61, div_14, sum_45, convolution_80, squeeze_196, relu_62, convolution_81, squeeze_199, relu_63, convolution_82, squeeze_202, relu_64, mean_15, convolution_83, relu_65, div_15, sum_48, convolution_85, squeeze_208, relu_66, convolution_86, squeeze_211, relu_67, convolution_87, squeeze_214, relu_68, mean_16, convolution_88, relu_69, div_16, sum_51, convolution_90, squeeze_220, relu_70, convolution_91, squeeze_223, relu_71, convolution_92, squeeze_226, relu_72, mean_17, convolution_93, relu_73, div_17, sum_54, convolution_95, squeeze_232, relu_74, convolution_96, squeeze_235, relu_75, convolution_97, squeeze_238, relu_76, mean_18, convolution_98, relu_77, div_18, sum_57, convolution_100, squeeze_244, relu_78, convolution_101, squeeze_247, relu_79, convolution_102, squeeze_250, relu_80, mean_19, convolution_103, relu_81, div_19, sum_60, convolution_105, squeeze_256, relu_82, convolution_106, squeeze_259, relu_83, convolution_107, squeeze_262, relu_84, mean_20, convolution_108, relu_85, div_20, sum_63, convolution_110, squeeze_268, relu_86, convolution_111, squeeze_271, relu_87, convolution_112, squeeze_274, relu_88, mean_21, convolution_113, relu_89, div_21, sum_66, convolution_115, squeeze_280, relu_90, convolution_116, squeeze_283, relu_91, convolution_117, squeeze_286, relu_92, mean_22, convolution_118, relu_93, div_22, sum_69, convolution_120, squeeze_292, relu_94, convolution_121, squeeze_295, relu_95, convolution_122, squeeze_298, relu_96, mean_23, convolution_123, relu_97, div_23, sum_72, convolution_125, squeeze_304, relu_98, convolution_126, squeeze_307, relu_99, convolution_127, squeeze_310, relu_100, mean_24, convolution_128, relu_101, div_24, sum_75, convolution_130, squeeze_316, relu_102, convolution_131, squeeze_319, relu_103, convolution_132, squeeze_322, relu_104, mean_25, convolution_133, relu_105, div_25, sum_78, convolution_135, squeeze_328, relu_106, convolution_136, squeeze_331, relu_107, convolution_137, squeeze_334, relu_108, mean_26, convolution_138, relu_109, div_26, sum_81, convolution_140, squeeze_340, relu_110, convolution_141, squeeze_343, relu_111, convolution_142, squeeze_346, relu_112, mean_27, convolution_143, relu_113, div_27, sum_84, convolution_145, squeeze_352, relu_114, convolution_146, squeeze_355, relu_115, convolution_147, squeeze_358, relu_116, mean_28, convolution_148, relu_117, div_28, sum_87, convolution_150, squeeze_364, relu_118, convolution_151, squeeze_367, relu_119, convolution_152, squeeze_370, relu_120, mean_29, convolution_153, relu_121, div_29, sum_90, convolution_155, squeeze_376, relu_122, convolution_156, squeeze_379, relu_123, convolution_157, squeeze_382, relu_124, mean_30, convolution_158, relu_125, div_30, sum_93, avg_pool2d_4, convolution_160, squeeze_388, avg_pool2d_5, convolution_161, squeeze_391, relu_126, convolution_162, squeeze_394, relu_127, convolution_163, squeeze_397, relu_128, mean_31, convolution_164, relu_129, div_31, sum_96, convolution_166, squeeze_403, relu_130, convolution_167, squeeze_406, relu_131, convolution_168, squeeze_409, relu_132, mean_32, convolution_169, relu_133, div_32, sum_99, convolution_171, squeeze_415, view_198, permute_34, le, unsqueeze_558, unsqueeze_584, unsqueeze_596, unsqueeze_608, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_696, unsqueeze_708, unsqueeze_720, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_796, unsqueeze_808, unsqueeze_820, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_896, unsqueeze_908, unsqueeze_920, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_996, unsqueeze_1008, unsqueeze_1020, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1096, unsqueeze_1108, unsqueeze_1120, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, unsqueeze_1196, unsqueeze_1208, unsqueeze_1220, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, unsqueeze_1296, unsqueeze_1308, unsqueeze_1320, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1396, unsqueeze_1408, unsqueeze_1420, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1496, unsqueeze_1508, unsqueeze_1520, unsqueeze_1546, unsqueeze_1558, unsqueeze_1570, unsqueeze_1596, unsqueeze_1608, unsqueeze_1620, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, unsqueeze_1696, unsqueeze_1708, unsqueeze_1720, unsqueeze_1746, unsqueeze_1758, unsqueeze_1770, unsqueeze_1796, unsqueeze_1808, unsqueeze_1820, unsqueeze_1832, unsqueeze_1858, unsqueeze_1870, unsqueeze_1882, unsqueeze_1908, unsqueeze_1920, unsqueeze_1932, unsqueeze_1958, unsqueeze_1970, unsqueeze_1982, unsqueeze_2008, unsqueeze_2020, unsqueeze_2032, unsqueeze_2044, unsqueeze_2070, unsqueeze_2082, unsqueeze_2094, unsqueeze_2120, unsqueeze_2132, unsqueeze_2144, unsqueeze_2170, unsqueeze_2182, unsqueeze_2194, unsqueeze_2206, unsqueeze_2232, unsqueeze_2244, unsqueeze_2256, unsqueeze_2268, unsqueeze_2280, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnest101e', benchmark_compiled_module)
