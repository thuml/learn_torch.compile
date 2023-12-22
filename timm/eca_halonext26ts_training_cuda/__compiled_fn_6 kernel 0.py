
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtcbt4g57hrviadj7jcv3toxxh4xcnjikjw4nylbesjf4s5onol.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': []}
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ma/cmakcptylie6ti647xtggg7jr34pk44pv4b2jdk67do6outbbog7.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_div_mul_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_native_batch_norm_backward_1', 'mutated_arg_names': []}
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
    r2 = (rindex // 64)
    x0 = xindex
    r1 = rindex % 64
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3lmrudbqurjkb6unwzp343mhbdkaznzg3umtnaczy5xzqpkmgg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 64)
    x4 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None)
    tmp5 = tl.load(in_ptr2 + (x4), None)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
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
    tl.store(out_ptr0 + (x4), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3ktmjstbb43sqn62qxzoimfwffqgyjap6qejjtmrxtuqd72otz.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caetjatq6bnxcvif3lmkzjvxeiaxir3qwtqrt3ib2mqslzsdmoyj.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_mul_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_native_batch_norm_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = 0.001953125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp2 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6ym32vldzrrke3morwh7x2s2zr5v6b742mne32fq577ztyckso6.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vw752bswzez4h7kncxuugosraqngiccgmlypse3p6rzqvg4dn7.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_per_fused__softmax_backward_data_mul_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r4 = rindex
    x5 = xindex
    x6 = (xindex // 12)
    x0 = xindex % 12
    x1 = (xindex // 12) % 8
    x2 = (xindex // 96) % 8
    x3 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (r4 + (12*x5)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r4 + (12*x5)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (12*r4) + (144*x6)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + (12*r4) + (144*x6)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 * tmp11
    tmp13 = tmp11 * tmp3
    tmp14 = tmp12 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.25
    tmp20 = tmp5 * tmp19
    tl.store(out_ptr2 + (r4 + (12*x5)), tmp20, rmask)
    tl.store(out_ptr0 + (x0 + (12*x2) + (96*x1) + (768*x3)), tmp9, None)
    tl.store(out_ptr1 + (x5), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zr/czrxdymtrecpxj7opczctk3xcsut236srhnyaeooy2y44ctnmn2w.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 23
    x1 = (xindex // 23)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 24, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (24*(x1 % 8))
    tmp4 = tl.full([1], 207, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (24*(x1 % 8))) // 23) % 9
    tmp8 = tl.full([1], 8, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (24*(x1 % 8))) % 23
    tmp12 = tl.full([1], 11, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-11) + (12*(((x0 + (24*(x1 % 8))) // 23) % 9)) + (96*(x1 // 8)) + ((x0 + (24*(x1 % 8))) % 23)), tmp14, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp21, tmp18)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp6, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47g3he7mbifzglrytmma6nxsc4j2jr7jcp6ujxnbd72tjimjupp.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 737280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civcyp6ishnubegexdplui5k3gkewzrttlqlek3tasyazwvnqhd7.py
# Source Nodes: [], Original ATen: [aten.clone, aten.unfold_backward]

triton_poi_fused_clone_unfold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_unfold_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 80
    x2 = xindex
    y1 = (yindex // 80)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (144*y0) + (2304*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 80, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-16) + y0 + (64*x2) + (9216*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (144*y3)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvf27s4eueo64w35mucag3rdn2ku5t6oroli4txc2d4pkjcvuzm.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16], 
    filename=__file__,
    triton_meta={'signature': {0: '*i32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kme6vuhnvhgsxdgykodzlicqmymwyl7b4qthrnohgyq2in737t.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]

triton_poi_fused_constant_pad_nd_convolution_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_backward_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 8) % 8
    x0 = xindex % 8
    x2 = (xindex // 64)
    x4 = xindex
    tmp0 = 2 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 2 + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (26 + x0 + (12*x1) + (144*x2)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/me/cmeufaerrakqklizvdsgojjfosbcfsbr7ygoqjhp737w5ts56i4f.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 8
    y1 = (yindex // 8) % 8
    y2 = (yindex // 64)
    y4 = yindex % 64
    tmp0 = tl.load(in_ptr0 + ((16*y1) + (128*y0) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((16*y0) + (128*y1) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*y0) + (128*y1) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y4 + (64*x3) + (8192*y2)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clpfpwryogwy4bu7kmjhcyhogvydgb4exh25vldjg5vahz74rbcf.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_mul_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_batch_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/nc/cncjredc7vhutmo2hiwnrhp7dx4qeo5qwk4gsr3i4h7bpnrg462f.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/jp/cjphsv4y5gvcat7m26fitmh72xblppngb3acncfh5vzcnebbnmdj.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_mul_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = (rindex // 64)
    x0 = xindex
    r1 = rindex % 64
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 - tmp14
    tmp16 = tmp8 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = tmp21 - tmp22
    tmp24 = tmp8 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp20 * tmp29
    tmp32 = tmp28 * tmp31
    tl.store(out_ptr3 + (x0), tmp30, None)
    tl.store(out_ptr4 + (x0), tmp32, None)
    tl.store(out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (x0), tmp20, None)
    tl.store(out_ptr2 + (x0), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2tyjm2orfyworqb4pde76eqx5o7mhwsjae7zrpvi6ak2d6wezd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 64)
    x4 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None)
    tmp5 = tl.load(in_ptr2 + (x4), None)
    tmp7 = tl.load(in_ptr3 + (x4), None)
    tmp9 = tl.load(in_ptr4 + (x4), None)
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x4), None)
    tmp24 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 - tmp10
    tmp13 = 0.001953125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp25 = tmp23 - tmp24
    tmp27 = tmp26 * tmp13
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp25 * tmp30
    tmp32 = tmp8 - tmp31
    tmp33 = tmp32 - tmp21
    tmp35 = tmp15 * tmp34
    tmp36 = tmp22 * tmp35
    tmp38 = tmp28 * tmp37
    tmp39 = tmp33 * tmp38
    tl.store(in_out_ptr0 + (x4), tmp36, None)
    tl.store(in_out_ptr1 + (x4), tmp39, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzf6yvk4eqdtv74cqhbt4cvcjiuoms3eybcb7dlbe5mjrqzuwfz.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 16
    x3 = (xindex // 16)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + ((4*(x3 % 2)) + (8*(x2 // 4)) + (32*(x3 // 2)) + (64*y4) + (x2 % 4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((4*(x3 % 2)) + (8*(x2 // 4)) + (32*(x3 // 2)) + (64*y4) + (x2 % 4)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((4*(x3 % 2)) + (8*(x2 // 4)) + (32*(x3 // 2)) + (64*y4) + (x2 % 4)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y4 % 512), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y4 % 512), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y4 % 512), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y4 % 512), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = 0.001953125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp2 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (y0 + (64*x5) + (4096*y1)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cecucn3o4f2u3dqnblvsaqknps7hnm3bbab4djdjlplteexbrt3h.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_per_fused__softmax_backward_data_mul_sum_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r4 = rindex
    x5 = xindex
    x6 = (xindex // 12)
    x0 = xindex % 12
    x1 = (xindex // 12) % 4
    x2 = (xindex // 48) % 4
    x3 = (xindex // 192)
    tmp0 = tl.load(in_ptr0 + (r4 + (12*x5)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r4 + (12*x5)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (12*r4) + (144*x6)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + (12*r4) + (144*x6)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 * tmp11
    tmp13 = tmp11 * tmp3
    tmp14 = tmp12 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.25
    tmp20 = tmp5 * tmp19
    tl.store(out_ptr2 + (r4 + (12*x5)), tmp20, rmask)
    tl.store(out_ptr0 + (x0 + (12*x2) + (48*x1) + (192*x3)), tmp9, None)
    tl.store(out_ptr1 + (x5), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddbkci4tkz6mqawobgnq3uwn4qxmn2uwm7ctnu4jji326ge767e.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 23
    x1 = (xindex // 23)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 24, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (24*(x1 % 4))
    tmp4 = tl.full([1], 115, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (24*(x1 % 4))) // 23) % 5
    tmp8 = tl.full([1], 4, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (24*(x1 % 4))) % 23
    tmp12 = tl.full([1], 11, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-11) + (12*(((x0 + (24*(x1 % 4))) // 23) % 5)) + (48*(x1 // 4)) + ((x0 + (24*(x1 % 4))) % 23)), tmp14, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp21, tmp18)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp6, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5hdyznwlc2agnowqqxqoc2lohvjmyckgaz34et4ngh3p243oe2.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2457600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5hdh2g324rmrevo5ti3nznc7qcjpjedoxa3wsqjby67uuord4q.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 245760
    xnumel = 12
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 12
    y1 = (yindex // 12) % 640
    y2 = (yindex // 7680) % 4
    y3 = (yindex // 30720)
    tmp0 = ((x4 + (12*y0) + (144*y2) + (576*y1)) // 576) % 80
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (12*y0) + (144*(((x4 + (12*y0) + (144*y2) + (576*y1)) // 576) % 80)) + (2304*y2) + (2304*((x4 + (12*y0)) // 144)) + (9216*((x4 + (12*y0) + (144*y2) + (576*y1)) // 46080)) + (73728*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 80, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-16) + (64*x4) + (768*y0) + (9216*y2) + (9216*((x4 + (12*y0)) // 144)) + (36864*((x4 + (12*y0) + (144*y2) + (576*y1)) // 46080)) + (294912*y3) + (((x4 + (12*y0) + (144*y2) + (576*y1)) // 576) % 80)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (12*x4) + (144*y2) + (576*y1) + (368640*y3)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bly6y3ydf7tku6m3d4sb7uc7usflnkiopiqthhcdme26mawu7o.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*i32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 12
    x1 = (xindex // 12)
    x2 = xindex
    tmp0 = x0 + (8*x1)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmdnxgmmjxxwe74ppqlrd5773j7klgufqywx6p5gtqkgkmsytbi.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cype6urdxrzbbcqurhzn7545paxf3wffgf42mtls7jrbwuvss464.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 122880
    xnumel = 20
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 12
    y1 = (yindex // 12)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (12*x2) + (240*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (20*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7udwirvjzhtuhx55iz4kr25shh53kvpugzarssfyqgp6lnebtwu.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]

triton_poi_fused_constant_pad_nd_convolution_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16) % 16
    x0 = xindex % 16
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = 2 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 2 + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (42 + x0 + (20*x1) + (400*x2)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5fnft6rbvtck7f25blbb7ksnekkjvn5ix6r4dsmrkaxiruxl4h.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 8
    y1 = (yindex // 8) % 8
    y2 = (yindex // 64)
    y4 = yindex % 64
    tmp0 = tl.load(in_ptr0 + ((16*((((4*(y1 % 4)) + (y0 % 4)) // 4) % 4)) + (64*y0) + (512*((y0 + (8*y1)) // 32)) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((16*(y0 % 4)) + (64*((((4*(y1 % 4)) + (y0 % 4)) // 4) % 4)) + (256*(y0 // 4)) + (512*((y0 + (8*y1)) // 32)) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*(y0 % 4)) + (64*(y1 % 4)) + (256*(y0 // 4)) + (512*((y0 + (8*y1)) // 32)) + (1024*((y0 + (8*y1) + (64*x3)) // 1024)) + (8192*y2) + (((y0 + (8*y1) + (64*x3)) // 64) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y4 + (64*x3) + (8192*y2)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgivjpiem5yfazggdftyr734mdetyy7rmn3wsbeiybtecgoigtpw.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
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
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gh/cghps4iipffrckp5akma6ejcsy7coqqrwye7gf6hminpesejxgna.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkvs7labb6r2sazduk4i4yhon6ui5dnij4hvgcxfzxrhq3r2ykr.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_29', 'mutated_arg_names': []}
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
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
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
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co75r2gfudiy4ftx4ov2p6xd4dvohezw76my3y3ccblxxa6va5dc.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_30', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
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
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndnc64iqe2onki3zypy6zw3y67cqrsr6vcfw5gv5fwkkpjnpb3u.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgizi3nyhklqyxq66s25nzhnkjy67dg27hpaoqebl7jjlynv6x3u.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + ((8*(x3 % 2)) + (16*(x2 // 8)) + (128*(x3 // 2)) + (256*y4) + (x2 % 8)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((8*(x3 % 2)) + (16*(x2 // 8)) + (128*(x3 // 2)) + (256*y4) + (x2 % 8)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((8*(x3 % 2)) + (16*(x2 // 8)) + (128*(x3 // 2)) + (256*y4) + (x2 % 8)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y4 % 256), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y4 % 256), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y4 % 256), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y4 % 256), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp2 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (y0 + (32*x5) + (8192*y1)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdxdjavqp3t75soxnosld3x52ukanbktje67hnlocri6r3ut7xc.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyinsn7tvosweaimbbwcs53cy5yjzq3f4kvgfen7v3f7uo7p3aq7.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_per_fused__softmax_backward_data_mul_sum_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[262144, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r4 = rindex
    x5 = xindex
    x6 = (xindex // 12)
    x0 = xindex % 12
    x1 = (xindex // 12) % 8
    x2 = (xindex // 96) % 8
    x3 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (r4 + (12*x5)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r4 + (12*x5)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (12*r4) + (144*x6)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + (12*r4) + (144*x6)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 * tmp11
    tmp13 = tmp11 * tmp3
    tmp14 = tmp12 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.25
    tmp20 = tmp5 * tmp19
    tl.store(out_ptr2 + (r4 + (12*x5)), tmp20, rmask)
    tl.store(out_ptr0 + (x0 + (12*x2) + (96*x1) + (768*x3)), tmp9, None)
    tl.store(out_ptr1 + (x5), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jdpiaowapf7vd754dymqywmua2jbgvwui5tfovfqxjffb3hbsv.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 23
    x1 = (xindex // 23)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 24, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (24*(x1 % 8))
    tmp4 = tl.full([1], 207, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (24*(x1 % 8))) // 23) % 9
    tmp8 = tl.full([1], 8, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (24*(x1 % 8))) % 23
    tmp12 = tl.full([1], 11, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-11) + (12*(((x0 + (24*(x1 % 8))) // 23) % 9)) + (96*(x1 // 8)) + ((x0 + (24*(x1 % 8))) % 23)), tmp14, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp21, tmp18)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp6, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kf/ckfe2vbh5n7qvpowz34zepfotwylobu4omiabtf5ouhyndjtys5t.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1474560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hdydpsdbwizmgizl7nzhnd7irjgnl54rt5jdk7usedruvrdfbb.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
    xnumel = 12
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 12
    y1 = (yindex // 12) % 384
    y2 = (yindex // 4608) % 4
    y3 = (yindex // 18432)
    tmp0 = ((x4 + (12*y0) + (144*y2) + (576*y1)) // 576) % 48
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (12*y0) + (144*(((x4 + (12*y0) + (144*y2) + (576*y1)) // 576) % 48)) + (2304*y2) + (2304*((x4 + (12*y0)) // 144)) + (9216*((x4 + (12*y0) + (144*y2) + (576*y1)) // 27648)) + (73728*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 48, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-16) + (32*x4) + (384*y0) + (4608*y2) + (4608*((x4 + (12*y0)) // 144)) + (18432*((x4 + (12*y0) + (144*y2) + (576*y1)) // 27648)) + (147456*y3) + (((x4 + (12*y0) + (144*y2) + (576*y1)) // 576) % 48)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (12*x4) + (144*y2) + (576*y1) + (221184*y3)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnb2prja4yr7c6sgm7hecxwuuh7ki5zzp5juy65cnbny3o5s6a6.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1228800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7ljevjqduzmxi5kcbnc7v2qx2bggpi2kaggmdzgoub4spns262.py
# Source Nodes: [], Original ATen: [aten.unfold_backward]

triton_poi_fused_unfold_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 20
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 12
    y1 = (yindex // 12)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (12*x2) + (240*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (20*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwf4aqikz3uwbbmsjfahsyfluqfkmwkpavy2tsub26qa7ala6op.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]

triton_poi_fused_constant_pad_nd_convolution_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16) % 16
    x0 = xindex % 16
    x2 = (xindex // 256)
    x4 = xindex
    tmp0 = 2 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 2 + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (42 + x0 + (20*x1) + (400*x2)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbjhfc4jqjjyanotz2ygljpthc3rerna3csu6tg65sfrwlxohku.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 16
    y2 = (yindex // 256)
    y4 = yindex % 256
    tmp0 = tl.load(in_ptr0 + ((16*((((8*(y1 % 8)) + (y0 % 8)) // 8) % 8)) + (128*y0) + (2048*((y0 + (16*y1)) // 128)) + (4096*((y0 + (16*y1) + (256*x3)) // 4096)) + (32768*y2) + (((y0 + (16*y1) + (256*x3)) // 256) % 16)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((16*(y0 % 8)) + (128*((((8*(y1 % 8)) + (y0 % 8)) // 8) % 8)) + (1024*(y0 // 8)) + (2048*((y0 + (16*y1)) // 128)) + (4096*((y0 + (16*y1) + (256*x3)) // 4096)) + (32768*y2) + (((y0 + (16*y1) + (256*x3)) // 256) % 16)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*(y0 % 8)) + (128*(y1 % 8)) + (1024*(y0 // 8)) + (2048*((y0 + (16*y1)) // 128)) + (4096*((y0 + (16*y1) + (256*x3)) // 4096)) + (32768*y2) + (((y0 + (16*y1) + (256*x3)) // 256) % 16)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (y4 + (256*x3) + (32768*y2)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqscxswnonp6rkpopkt3jmwixxwy5lpcbse3nnhxnquq4ehhkca.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
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
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curu54yotdpzfi7qoqbr6fmdge4x3hojhpw2sclbee6xhix5yr6i.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/up/cupbi454hto4vmivvrqffmsgwiqqrgbriaujv5hw7wl6bi24skmq.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_44', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpotejykupjl2grodlp32svsipb62oxu7fmplecmogzslt5ivdg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ik/cika2pb2yscnx266krwzyp576g6m2zqiqmy4hblyeupezlezgz6f.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr7 + (x3), None)
    tmp19 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/za/czalku7lxalpqs5opubfsf2bhlwmdjcque652ti63vb74hb5v6m2.py
# Source Nodes: [sigmoid_4, x_129], Original ATen: [aten.convolution_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# sigmoid_4 => sigmoid_21
# x_129 => mul_153, sigmoid_20
triton_per_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_47', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp8 * tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7bkujal7hbcsune62lbfo7otutrnsmweffzqnm5rqbyysn5qvs.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]

triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 256.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
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
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr2 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/cit3x4mdtpw3at5eutqug4b3gk6rgfofqy46s6bz44hpc5e7r26n.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]

triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 256)
    x1 = (xindex // 256) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp16 = tl.load(in_ptr3 + (x3), None)
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
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
    tl.store(in_out_ptr0 + (x3), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4wv4a3w2kannyg6j636sifxhsbpg2xl7xtqkcb7g5uli76wvocm.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_50', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oo/coouzcfdg5yxuovxqrokyqoao6ixcznzc6xb52ocfdvvbvr7dyfr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_51', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0001220703125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czuu6ipipudabf5lhoxxngowxidi5gdmverbaw3ie47bjvrzkpep.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_52', 'mutated_arg_names': []}
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
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
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
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cobhw5wbqpxdasxrfrch3zji6nsiysnbibmte7cfmxr55njnx37q.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_53', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
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
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g3/cg3inj3luugn5ielxni3sjktnyiaxer7f4o3jsuops6omszpclel.py
# Source Nodes: [sigmoid_3, x_106], Original ATen: [aten.convolution_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# sigmoid_3 => sigmoid_17
# x_106 => mul_128, sigmoid_16
triton_per_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
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
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp8 * tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7trdawzriys6zagj4bgvls2u6nmyxlv57i7ravvcinlerjkuzhv.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]

triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 1024.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
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
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr2 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ab/cabhjzqqntci54a7tibtbpq5yghz23xnr2ogsrykrow26q55bghb.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]

triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 1024)
    x1 = (xindex // 1024) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp16 = tl.load(in_ptr3 + (x3), None)
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
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
    tl.store(in_out_ptr0 + (x3), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlhvrtnq6e6eiuxrd4zxfvofhnx3t6ynrimnmbr6drurjwrevhw.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_57', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4m/c4mwnvpxgw5ijkci25gaibknxx3gx3j6opiedgxiqhkhiuluu3eb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_58', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0001220703125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6f/c6f3ci2dajf4mfgfmtjwzk5kgkctri7us2rwpmsvgere3niiacse.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5wuqeq5bdnzghdegt7mhvnap35am6eb4eom7p3esjfeuuwc6he.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_60', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfobculjsdjzvqsfipycek5dsj7efqv4nwmmol5sjcfliwet4dk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_61', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tt/cttrd3f4evcmzbzdwjkvdcrx6vw6mk63ukg7rebqf7vgfj7w7emm.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_62', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c333dob42l3q6ndn26bnoozv4gklsrbajfgglv74bwlf45pht3to.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_63', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czvtt7g4tyzjxoqrvvju7c3zpeidy7azn6esvanjun5jdxbp4ie6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_64', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2jkl7kiaiymn4reiljhi5q2ibfco53siis4aetrncw6xpccdur.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_65', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.0517578125e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmk5fahvgnfbznbpypaxopubc5ybsukfiyhenzgsx2bthyhex75.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_66', 'mutated_arg_names': []}
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
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
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
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygps5wwo2w2crosarzztye5jaaxrhikwxautluxuz5bczldybvj.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
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
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pijb5olgug7xur47qmheitzhezn523aa5wrxwog6hslqcl4fum.py
# Source Nodes: [sigmoid_1, x_55], Original ATen: [aten.convolution_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# sigmoid_1 => sigmoid_9
# x_55 => mul_71, sigmoid_8
triton_red_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_68', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vd/cvda6onr6ujt3icnq4nysbqvzupgt4vjqxjl3y3pv6y5ysl75wep.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]

triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (64*(r2 // 4096)) + (128*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (64*(r2 // 4096)) + (128*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 4096.0
        tmp6 = tmp4 / tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = 1.0
        tmp11 = tmp10 - tmp9
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12 + tmp10
        tmp14 = tmp9 * tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ci/ccit7kvp7oirtfwo5lv35eesyhdrcmjg3paf3xquyvxhubhxpjst.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]

triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pw/cpws2t2cegx6gdu2glsjoi62zdm4hjgyms33idulxtqrvcc2gkx7.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]

triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ey/ceysixcstvdhy4nuzqbzzt2mdlozipksyg57d6bkcyd74jhobufm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]

triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_72', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 4096)
    x1 = (xindex // 4096) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp16 = tl.load(in_ptr3 + (x3), None)
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 4096.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 * tmp14
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
    tl.store(in_out_ptr0 + (x3), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7ykv54l6mlz4fern23gjfj76r7qirogytiyopu47tjvsfltmlf.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_73', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcpg2f7qcervs5t7523gzwft735g3ccedjtohrnvsjliubbnu5z.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.0517578125e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5p/c5pqwbvwyqnxz352ugh7tuznxer5sgbnbgqvgl7txonkzmrl3y2q.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xt/cxts4bnt3s6kddybppvvttzdpw2zy67rzkvbquptwwx4oabxkznz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/aq/caqh7eiklfspavm5envsqcmemwakl752zzsoq6vndncfb4zjgfy7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr7 + (x3), None)
    tmp19 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/oh/cohq5mevreozphnyj3wp5z6bgfkix3xsmvh4ozq4jlhpsarifenx.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_78', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/ciifbafcnyjsu32mxdhubjddxgsbjjtokw3inxsqyramnrvwotpf.py
# Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]

triton_poi_fused_add_max_pool2d_with_indices_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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


# kernel path: /tmp/torchinductor_youkaichao/my/cmydh5fom74kezwersnz4dqcmoac6zgqbbxclpkmhigzhvswvh5p.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_80', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (1048576*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (1048576*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (1048576*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jv5hbkbx6xy7ber3ypnljfb2m4uzudsqocdetwgorih5ssve3v.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_81', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxmskw77ymq7rqn64dx4xnjbtmlbdakc4a2emqzxsegoqlg3v76.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_82', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/io/ciopfnyndyaisakyvo4vsa7dnsqsumnpfi7fqepl7pdlr3a7lvbm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_83', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 7.62939453125e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvaeeytsvj5s7u6pchtfdsgny4ay7efk6nbn2dqnwhv64qz6o7uj.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyrmsgwwp5b6chnsqdqt3jolexq4d5ddyajdskedteamscr4bnl.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/qo/cqozgbm7khxkqrwmaw6cvz3nokhxvcdoroqm6i6veswi23cf6sul.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/5f/c5flz6iirbw6ljyjnmp2uxeltnbxtn4h6dffw4ygkic5thu7dzj4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_87', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 7.62939453125e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgcinzjznjdldtflml25ink2uxnmdt2xeqfny4zq2prsnlnw2p4.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (393216*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (393216*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (393216*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/po/cpo6ijxbjnm5yuruqilxihuprrlgvox22lgcb2xeewoik35fnxg7.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
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


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4bb2s3jwxfsfgjqoon6kkgfvkcekduilewt5f7kkno7lmdxp7i.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
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


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7t2yxlblebhinmzppknuvhatypxker4sgziupo2tvzpos43fqs.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_91', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 24
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 7.62939453125e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_203, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem_6, getitem_7, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, view_17, view_23, squeeze_67, mul_186, convolution_29, squeeze_70, mul_194, convolution_30, squeeze_73, mul_202, view_42, view_48, squeeze_76, mul_211, convolution_33, squeeze_79, convolution_34, squeeze_82, mul_226, convolution_35, squeeze_85, mul_234, view_67, view_73, squeeze_88, mul_243, convolution_38, squeeze_91, clone_51, permute_34, mul_253, unsqueeze_126, mul_265, sub_40, permute_42, permute_43, alias_8, permute_47, permute_53, permute_55, permute_56, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, sub_60, permute_68, permute_69, alias_9, permute_73, permute_79, permute_81, permute_82, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, sub_76, permute_94, permute_95, alias_10, permute_99, permute_105, permute_107, permute_108, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (24, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_57, (2048, ), (1, ))
    assert_size_stride(primals_59, (2048, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_67, (2048, ), (1, ))
    assert_size_stride(primals_69, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_70, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_71, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_72, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_74, (1, 1, 3), (3, 3, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_77, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_78, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_79, (1, 1, 3), (3, 3, 1))
    assert_size_stride(primals_80, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_81, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_82, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_83, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_84, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_85, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_86, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_87, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_88, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_89, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_90, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_91, (256, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_92, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_93, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_94, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_95, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_96, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_97, (384, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_99, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_100, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_102, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_103, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_105, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_106, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_107, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_203, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(convolution, (8, 24, 128, 128), (393216, 16384, 128, 1))
    assert_size_stride(squeeze_1, (24, ), (1, ))
    assert_size_stride(mul_7, (8, 24, 128, 128), (393216, 16384, 128, 1))
    assert_size_stride(convolution_1, (8, 32, 128, 128), (524288, 16384, 128, 1))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(mul_15, (8, 32, 128, 128), (524288, 16384, 128, 1))
    assert_size_stride(convolution_2, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(mul_23, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(getitem_6, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(getitem_7, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_3, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(mul_31, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_4, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(add_24, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(view, (8, 1, 64), (64, 64, 1))
    assert_size_stride(convolution_5, (8, 1, 64), (64, 64, 1))
    assert_size_stride(mul_40, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_6, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_16, (256, ), (1, ))
    assert_size_stride(convolution_7, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(mul_55, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_8, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(mul_63, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(add_45, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(view_2, (8, 1, 64), (64, 64, 1))
    assert_size_stride(convolution_10, (8, 1, 64), (64, 64, 1))
    assert_size_stride(mul_72, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_11, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_28, (256, ), (1, ))
    assert_size_stride(mul_80, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_12, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(mul_88, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(convolution_13, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(add_61, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(view_4, (8, 1, 128), (128, 128, 1))
    assert_size_stride(convolution_14, (8, 1, 128), (128, 128, 1))
    assert_size_stride(mul_97, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_15, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_37, (512, ), (1, ))
    assert_size_stride(convolution_16, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(mul_112, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(convolution_17, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(mul_120, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_18, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(add_82, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(view_6, (8, 1, 128), (128, 128, 1))
    assert_size_stride(convolution_19, (8, 1, 128), (128, 128, 1))
    assert_size_stride(mul_129, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(convolution_20, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(squeeze_49, (512, ), (1, ))
    assert_size_stride(mul_137, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(convolution_21, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(squeeze_52, (256, ), (1, ))
    assert_size_stride(mul_145, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(convolution_22, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_55, (256, ), (1, ))
    assert_size_stride(add_98, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(view_8, (8, 1, 256), (256, 256, 1))
    assert_size_stride(convolution_23, (8, 1, 256), (256, 256, 1))
    assert_size_stride(mul_154, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_24, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_58, (1024, ), (1, ))
    assert_size_stride(convolution_25, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_61, (1024, ), (1, ))
    assert_size_stride(mul_169, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_26, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(squeeze_64, (256, ), (1, ))
    assert_size_stride(mul_177, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(view_17, (16384, 16), (16, 1))
    assert_size_stride(view_23, (16384, 16), (16, 1))
    assert_size_stride(squeeze_67, (256, ), (1, ))
    assert_size_stride(mul_186, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_29, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(squeeze_70, (1024, ), (1, ))
    assert_size_stride(mul_194, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(convolution_30, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_73, (512, ), (1, ))
    assert_size_stride(mul_202, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(view_42, (4096, 16), (16, 1))
    assert_size_stride(view_48, (4096, 16), (16, 1))
    assert_size_stride(squeeze_76, (512, ), (1, ))
    assert_size_stride(mul_211, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(convolution_33, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(squeeze_79, (2048, ), (1, ))
    assert_size_stride(convolution_34, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(squeeze_82, (2048, ), (1, ))
    assert_size_stride(mul_226, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(convolution_35, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(squeeze_85, (512, ), (1, ))
    assert_size_stride(mul_234, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(view_67, (4096, 16), (16, 1))
    assert_size_stride(view_73, (4096, 16), (16, 1))
    assert_size_stride(squeeze_88, (512, ), (1, ))
    assert_size_stride(mul_243, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(convolution_38, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(squeeze_91, (2048, ), (1, ))
    assert_size_stride(clone_51, (8, 2048), (2048, 1))
    assert_size_stride(permute_34, (1000, 2048), (2048, 1))
    assert_size_stride(mul_253, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(unsqueeze_126, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(mul_265, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(sub_40, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(permute_42, (64, 144, 64), (9216, 1, 144))
    assert_size_stride(permute_43, (64, 64, 144), (9216, 144, 1))
    assert_size_stride(alias_8, (64, 1, 64, 144), (9216, 9216, 144, 1))
    assert_size_stride(permute_47, (23, 16), (16, 1))
    assert_size_stride(permute_53, (23, 16), (16, 1))
    assert_size_stride(permute_55, (64, 16, 64), (1024, 64, 1))
    assert_size_stride(permute_56, (64, 144, 16), (2304, 1, 144))
    assert_size_stride(mul_280, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(unsqueeze_150, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_292, (8, 2048, 8, 8), (131072, 64, 8, 1))
    assert_size_stride(unsqueeze_162, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_174, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(mul_313, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(sub_60, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(permute_68, (256, 144, 16), (2304, 1, 144))
    assert_size_stride(permute_69, (256, 64, 144), (9216, 1, 64))
    assert_size_stride(alias_9, (64, 4, 16, 144), (9216, 2304, 144, 1))
    assert_size_stride(permute_73, (23, 16), (16, 1))
    assert_size_stride(permute_79, (23, 16), (16, 1))
    assert_size_stride(permute_81, (256, 16, 16), (256, 1, 16))
    assert_size_stride(permute_82, (256, 144, 16), (2304, 1, 144))
    assert_size_stride(mul_328, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(unsqueeze_198, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_340, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(unsqueeze_210, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(mul_352, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(sub_76, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(permute_94, (256, 144, 64), (9216, 1, 144))
    assert_size_stride(permute_95, (256, 32, 144), (4608, 1, 32))
    assert_size_stride(alias_10, (64, 4, 64, 144), (36864, 9216, 144, 1))
    assert_size_stride(permute_99, (23, 16), (16, 1))
    assert_size_stride(permute_105, (23, 16), (16, 1))
    assert_size_stride(permute_107, (256, 16, 64), (1024, 1, 16))
    assert_size_stride(permute_108, (256, 144, 16), (2304, 1, 144))
    assert_size_stride(mul_367, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(unsqueeze_234, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_379, (8, 1024, 16, 16), (262144, 256, 16, 1))
    assert_size_stride(unsqueeze_246, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_272, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_416, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(unsqueeze_284, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_428, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(unsqueeze_296, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_456, (8, 128, 32, 32), (131072, 1024, 32, 1))
    assert_size_stride(unsqueeze_322, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_468, (8, 512, 32, 32), (524288, 1024, 32, 1))
    assert_size_stride(unsqueeze_334, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_360, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_505, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(unsqueeze_372, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_517, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(unsqueeze_384, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_398, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_545, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(unsqueeze_410, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_557, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(unsqueeze_422, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_448, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_594, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(unsqueeze_460, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_606, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(unsqueeze_472, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_618, (8, 32, 128, 128), (524288, 16384, 128, 1))
    assert_size_stride(unsqueeze_484, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_630, (8, 24, 128, 128), (393216, 16384, 128, 1))
    assert_size_stride(unsqueeze_496, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_34, out=buf0)
        del permute_34
        buf1 = empty((1000, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_51, out=buf1)
        del clone_51
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf4 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf5 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_1.run(buf0, mul_253, convolution_38, unsqueeze_126, squeeze_91, buf3, buf4, buf5, 2048, 512, grid=grid(2048), stream=stream0)
        buf6 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_2.run(buf0, mul_253, convolution_38, unsqueeze_126, buf4, squeeze_91, buf3, primals_67, buf6, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_38
        del primals_67
        del squeeze_91
        del unsqueeze_126
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        buf7 = aten.convolution_backward(buf6, mul_243, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_243
        del primals_107
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        buf10 = empty((512, ), device='cuda', dtype=torch.float32)
        buf11 = empty((512, ), device='cuda', dtype=torch.float32)
        buf12 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_3.run(buf8, mul_265, sub_40, squeeze_88, buf10, buf11, buf12, 512, 512, grid=grid(512), stream=stream0)
        buf13 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_mul_native_batch_norm_backward_4.run(buf13, mul_265, sub_40, buf11, squeeze_88, buf10, primals_65, 262144, grid=grid(262144), stream=stream0)
        del mul_265
        del primals_65
        del squeeze_88
        del sub_40
        buf14 = empty((64, 144, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_42, reinterpret_tensor(buf13, (64, 64, 64), (4096, 1, 64), 0), out=buf14)
        del permute_42
        buf15 = empty((64, 64, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (64, 64, 64), (4096, 1, 64), 0), permute_43, out=buf15)
        del permute_43
        buf16 = empty_strided((64, 1, 64, 1), (64, 4096, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_5.run(buf15, alias_8, buf16, 4096, 144, grid=grid(4096), stream=stream0)
        buf17 = empty((64, 8, 1, 8, 12), device='cuda', dtype=torch.float32)
        buf21 = empty((64, 8, 1, 8, 12), device='cuda', dtype=torch.float32)
        buf25 = empty((64, 1, 64, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_6.run(buf15, alias_8, buf16, buf17, buf21, buf25, 49152, 12, grid=grid(49152), stream=stream0)
        del alias_8
        del buf15
        buf18 = empty((4096, 23), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf17, buf18, 94208, grid=grid(94208), stream=stream0)
        buf19 = empty((23, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (23, 4096), (1, 23), 0), view_73, out=buf19)
        del view_73
        buf20 = empty((4096, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf18, permute_47, out=buf20)
        del permute_47
        buf22 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_7.run(buf21, buf22, 94208, grid=grid(94208), stream=stream0)
        buf23 = empty((23, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (23, 4096), (1, 23), 0), view_67, out=buf23)
        del view_67
        buf24 = empty((4096, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf22, permute_53, out=buf24)
        del permute_53
        buf26 = empty((64, 16, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_55, reinterpret_tensor(buf25, (64, 64, 144), (9216, 144, 1), 0), out=buf26)
        del permute_55
        buf27 = empty((64, 64, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf25, (64, 64, 144), (9216, 144, 1), 0), permute_56, out=buf27)
        del permute_56
        buf28 = empty((8, 640, 1, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_8.run(buf28, 737280, grid=grid(737280), stream=stream0)
        buf29 = empty((64, 80, 1, 144), device='cuda', dtype=torch.float32)
        buf30 = reinterpret_tensor(buf29, (8, 640, 1, 12, 12), (92160, 144, 737280, 1, 12), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.unfold_backward]
        triton_poi_fused_clone_unfold_backward_9.run(buf30, buf26, buf14, 5120, 144, grid=grid(5120, 144), stream=stream0)
        del buf26
        buf31 = empty((12, ), device='cuda', dtype=torch.int32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_10.run(buf31, 12, grid=grid(12), stream=stream0)
        aten.index_put_(buf28, [None, None, None, reinterpret_tensor(buf31, (12, ), (1, ), 0)], buf30, True)
        buf34 = reinterpret_tensor(buf30, (8, 640, 12, 12), (92160, 144, 12, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_8.run(buf34, 737280, grid=grid(737280), stream=stream0)
        aten.index_put_(buf34, [None, None, reinterpret_tensor(buf31, (12, ), (1, ), 0)], reinterpret_tensor(buf28, (8, 640, 12, 12), (92160, 144, 1, 12), 0), True)
        del buf28
        del buf31
        buf37 = empty((8, 640, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
        triton_poi_fused_constant_pad_nd_convolution_backward_11.run(buf34, buf37, 327680, grid=grid(327680), stream=stream0)
        del buf34
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
        buf38 = aten.convolution_backward(buf37, mul_234, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf37
        del primals_106
        buf39 = buf38[0]
        buf40 = buf38[1]
        del buf38
        buf41 = empty((8, 128, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_12.run(buf20, buf24, buf27, buf41, 512, 128, grid=grid(512, 128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf42 = aten.convolution_backward(buf41, mul_234, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_234
        del primals_105
        buf43 = buf42[0]
        buf44 = buf42[1]
        del buf42
        buf45 = buf11; del buf11  # reuse
        buf46 = empty((512, ), device='cuda', dtype=torch.float32)
        buf48 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_13.run(buf39, buf43, mul_280, convolution_35, unsqueeze_150, squeeze_85, buf45, buf46, buf48, 512, 512, grid=grid(512), stream=stream0)
        buf47 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_14.run(buf47, buf43, mul_280, convolution_35, unsqueeze_150, buf46, squeeze_85, buf45, primals_61, 262144, grid=grid(262144), stream=stream0)
        del convolution_35
        del mul_280
        del primals_61
        del squeeze_85
        del unsqueeze_150
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf49 = aten.convolution_backward(buf47, mul_226, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_226
        del primals_104
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        buf52 = buf4; del buf4  # reuse
        buf53 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf60 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf55 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf62 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_mul_native_batch_norm_backward_15.run(buf0, mul_253, buf50, mul_292, convolution_34, unsqueeze_162, convolution_33, unsqueeze_174, squeeze_82, squeeze_79, buf52, buf53, buf60, buf55, buf62, 2048, 512, grid=grid(2048), stream=stream0)
        buf54 = buf6; del buf6  # reuse
        buf61 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.float32)
        buf56 = buf54; del buf54  # reuse
        buf63 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_mul_native_batch_norm_backward_16.run(buf56, buf63, buf0, mul_253, buf50, mul_292, convolution_34, unsqueeze_162, buf53, squeeze_82, buf52, convolution_33, unsqueeze_174, buf60, squeeze_79, primals_59, primals_57, 1048576, grid=grid(1048576), stream=stream0)
        del buf50
        del buf53
        del convolution_33
        del convolution_34
        del mul_253
        del mul_292
        del primals_57
        del primals_59
        del squeeze_79
        del squeeze_82
        del unsqueeze_162
        del unsqueeze_174
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf57 = aten.convolution_backward(buf56, mul_194, primals_103, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf56
        del primals_103
        buf58 = buf57[0]
        buf59 = buf57[1]
        del buf57
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf64 = aten.convolution_backward(buf63, mul_211, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf63
        del mul_211
        del primals_102
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf46; del buf46  # reuse
        buf68 = empty((512, ), device='cuda', dtype=torch.float32)
        buf69 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_3.run(buf65, mul_313, sub_60, squeeze_76, buf67, buf68, buf69, 512, 512, grid=grid(512), stream=stream0)
        buf70 = reinterpret_tensor(buf47, (64, 4, 16, 64), (4096, 1024, 64, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf65, mul_313, sub_60, buf68, squeeze_76, buf67, primals_55, buf70, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del mul_313
        del primals_55
        del squeeze_76
        del sub_60
        buf71 = empty((256, 144, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_68, reinterpret_tensor(buf70, (256, 16, 64), (1024, 64, 1), 0), out=buf71)
        del permute_68
        buf72 = reinterpret_tensor(buf14, (256, 16, 144), (2304, 144, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (256, 16, 64), (1024, 64, 1), 0), permute_69, out=buf72)
        del permute_69
        buf73 = reinterpret_tensor(buf16, (64, 4, 16, 1), (64, 16, 1, 4096), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_5.run(buf72, alias_9, buf73, 4096, 144, grid=grid(4096), stream=stream0)
        buf74 = reinterpret_tensor(buf21, (256, 4, 1, 4, 12), (192, 48, 48, 12, 1), 0); del buf21  # reuse
        buf78 = reinterpret_tensor(buf17, (256, 4, 1, 4, 12), (192, 48, 48, 12, 1), 0); del buf17  # reuse
        buf82 = reinterpret_tensor(buf25, (64, 4, 16, 144), (9216, 2304, 144, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_18.run(buf72, alias_9, buf73, buf74, buf78, buf82, 49152, 12, grid=grid(49152), stream=stream0)
        del alias_9
        del buf73
        buf75 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf74, buf75, 94208, grid=grid(94208), stream=stream0)
        del buf74
        buf76 = empty((23, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (23, 4096), (1, 23), 0), view_48, out=buf76)
        del view_48
        buf77 = reinterpret_tensor(buf41, (4096, 16), (16, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf75, permute_73, out=buf77)
        del permute_73
        buf79 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf78, buf79, 94208, grid=grid(94208), stream=stream0)
        del buf78
        buf80 = empty((23, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (23, 4096), (1, 23), 0), view_42, out=buf80)
        del view_42
        buf81 = reinterpret_tensor(buf27, (4096, 16), (16, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf79, permute_79, out=buf81)
        del buf79
        del permute_79
        buf83 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_81, reinterpret_tensor(buf82, (256, 16, 144), (2304, 144, 1), 0), out=buf83)
        del permute_81
        buf84 = reinterpret_tensor(buf24, (256, 16, 16), (256, 16, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (256, 16, 144), (2304, 144, 1), 0), permute_82, out=buf84)
        del buf82
        del permute_82
        buf85 = empty((8, 640, 2, 20, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_20.run(buf85, 2457600, grid=grid(2457600), stream=stream0)
        buf86 = empty((8, 640, 2, 2, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_21.run(buf83, buf71, buf86, 245760, 12, grid=grid(245760, 12), stream=stream0)
        buf87 = empty((2, 12), device='cuda', dtype=torch.int32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_22.run(buf87, 24, grid=grid(24), stream=stream0)
        aten.index_put_(buf85, [None, None, None, reinterpret_tensor(buf87, (24, ), (1, ), 0)], reinterpret_tensor(buf86, (8, 640, 2, 24, 12), (368640, 576, 288, 12, 1), 0), True)
        del buf86
        buf90 = empty((8, 640, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_23.run(buf90, 2048000, grid=grid(2048000), stream=stream0)
        buf91 = empty((8, 640, 2, 12, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_24.run(buf85, buf91, 122880, 20, grid=grid(122880, 20), stream=stream0)
        del buf85
        aten.index_put_(buf90, [None, None, reinterpret_tensor(buf87, (24, ), (1, ), 0)], reinterpret_tensor(buf91, (8, 640, 24, 20), (307200, 480, 20, 1), 0), True)
        del buf91
        buf94 = empty((8, 640, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
        triton_poi_fused_constant_pad_nd_convolution_backward_25.run(buf90, buf94, 1310720, grid=grid(1310720), stream=stream0)
        del buf90
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
        buf95 = aten.convolution_backward(buf94, mul_202, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf94
        del primals_101
        buf96 = buf95[0]
        buf97 = buf95[1]
        del buf95
        buf98 = reinterpret_tensor(buf20, (8, 128, 8, 8), (8192, 64, 8, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_26.run(buf77, buf81, buf84, buf98, 512, 128, grid=grid(512, 128), stream=stream0)
        del buf77
        del buf81
        del buf84
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward(buf98, mul_202, primals_100, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf98
        del mul_202
        del primals_100
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf102 = buf68; del buf68  # reuse
        buf103 = empty((512, ), device='cuda', dtype=torch.float32)
        buf105 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_27.run(buf96, buf100, mul_328, convolution_30, unsqueeze_198, squeeze_73, buf102, buf103, buf105, 512, 2048, grid=grid(512), stream=stream0)
        buf104 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_28.run(buf104, buf96, mul_328, convolution_30, unsqueeze_198, buf103, squeeze_73, buf102, primals_51, 1048576, grid=grid(1048576), stream=stream0)
        del buf96
        del convolution_30
        del mul_328
        del primals_51
        del squeeze_73
        del unsqueeze_198
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf106 = aten.convolution_backward(buf104, mul_194, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf104
        del mul_194
        del primals_99
        buf107 = buf106[0]
        buf108 = buf106[1]
        del buf106
        buf109 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf110 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf112 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_29.run(buf58, buf107, mul_340, convolution_29, unsqueeze_210, squeeze_70, buf109, buf110, buf112, 1024, 2048, grid=grid(1024), stream=stream0)
        buf111 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_30.run(buf58, buf107, mul_340, convolution_29, unsqueeze_210, buf110, squeeze_70, buf109, primals_49, buf111, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_29
        del primals_49
        del squeeze_70
        del unsqueeze_210
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf113 = aten.convolution_backward(buf111, mul_186, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf111
        del mul_186
        del primals_98
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        buf116 = empty((256, ), device='cuda', dtype=torch.float32)
        buf117 = empty((256, ), device='cuda', dtype=torch.float32)
        buf118 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_31.run(buf114, mul_352, sub_76, squeeze_67, buf116, buf117, buf118, 256, 2048, grid=grid(256), stream=stream0)
        buf119 = empty((64, 4, 64, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf114, mul_352, sub_76, buf117, squeeze_67, buf116, primals_47, buf119, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf114
        del mul_352
        del primals_47
        del squeeze_67
        del sub_76
        buf120 = empty((256, 144, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_94, reinterpret_tensor(buf119, (256, 64, 32), (2048, 32, 1), 0), out=buf120)
        del permute_94
        buf121 = reinterpret_tensor(buf71, (256, 64, 144), (9216, 144, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (256, 64, 32), (2048, 32, 1), 0), permute_95, out=buf121)
        del buf119
        del permute_95
        buf122 = reinterpret_tensor(buf0, (64, 4, 64, 1), (256, 64, 1, 16384), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_33.run(buf121, alias_10, buf122, 16384, 144, grid=grid(16384), stream=stream0)
        buf123 = empty((256, 8, 1, 8, 12), device='cuda', dtype=torch.float32)
        buf127 = empty((256, 8, 1, 8, 12), device='cuda', dtype=torch.float32)
        buf131 = empty((64, 4, 64, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_34.run(buf121, alias_10, buf122, buf123, buf127, buf131, 196608, 12, grid=grid(196608), stream=stream0)
        del alias_10
        del buf121
        del buf122
        buf124 = empty((16384, 23), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_35.run(buf123, buf124, 376832, grid=grid(376832), stream=stream0)
        del buf123
        buf125 = empty((23, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (23, 16384), (1, 23), 0), view_23, out=buf125)
        del view_23
        buf126 = reinterpret_tensor(buf70, (16384, 16), (16, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf124, permute_99, out=buf126)
        del permute_99
        buf128 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_35.run(buf127, buf128, 376832, grid=grid(376832), stream=stream0)
        del buf127
        buf129 = empty((23, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (23, 16384), (1, 23), 0), view_17, out=buf129)
        del view_17
        buf130 = reinterpret_tensor(buf65, (16384, 16), (16, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf128, permute_105, out=buf130)
        del buf128
        del permute_105
        buf132 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_107, reinterpret_tensor(buf131, (256, 64, 144), (9216, 144, 1), 0), out=buf132)
        del permute_107
        buf133 = reinterpret_tensor(buf43, (256, 64, 16), (1024, 16, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf131, (256, 64, 144), (9216, 144, 1), 0), permute_108, out=buf133)
        del buf131
        del permute_108
        buf134 = empty((8, 384, 2, 20, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_36.run(buf134, 1474560, grid=grid(1474560), stream=stream0)
        buf135 = empty((8, 384, 2, 2, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_37.run(buf132, buf120, buf135, 147456, 12, grid=grid(147456, 12), stream=stream0)
        del buf120
        del buf132
        aten.index_put_(buf134, [None, None, None, reinterpret_tensor(buf87, (24, ), (1, ), 0)], reinterpret_tensor(buf135, (8, 384, 2, 24, 12), (221184, 576, 288, 12, 1), 0), True)
        del buf135
        buf138 = empty((8, 384, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_38.run(buf138, 1228800, grid=grid(1228800), stream=stream0)
        buf139 = empty((8, 384, 2, 12, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.unfold_backward]
        triton_poi_fused_unfold_backward_39.run(buf134, buf139, 73728, 20, grid=grid(73728, 20), stream=stream0)
        del buf134
        aten.index_put_(buf138, [None, None, reinterpret_tensor(buf87, (24, ), (1, ), 0)], reinterpret_tensor(buf139, (8, 384, 24, 20), (184320, 480, 20, 1), 0), True)
        del buf139
        del buf87
        buf142 = empty((8, 384, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
        triton_poi_fused_constant_pad_nd_convolution_backward_40.run(buf138, buf142, 786432, grid=grid(786432), stream=stream0)
        del buf138
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
        buf143 = aten.convolution_backward(buf142, mul_177, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf142
        del primals_97
        buf144 = buf143[0]
        buf145 = buf143[1]
        del buf143
        buf146 = reinterpret_tensor(buf13, (8, 128, 16, 16), (32768, 256, 16, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_41.run(buf126, buf130, buf133, buf146, 2048, 128, grid=grid(2048, 128), stream=stream0)
        del buf126
        del buf130
        del buf133
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf147 = aten.convolution_backward(buf146, mul_177, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf146
        del mul_177
        del primals_96
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf150 = buf117; del buf117  # reuse
        buf151 = empty((256, ), device='cuda', dtype=torch.float32)
        buf153 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_42.run(buf144, buf148, mul_367, convolution_26, unsqueeze_234, squeeze_64, buf150, buf151, buf153, 256, 2048, grid=grid(256), stream=stream0)
        buf152 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_43.run(buf152, buf148, mul_367, convolution_26, unsqueeze_234, buf151, squeeze_64, buf150, primals_43, 524288, grid=grid(524288), stream=stream0)
        del buf148
        del convolution_26
        del mul_367
        del primals_43
        del squeeze_64
        del unsqueeze_234
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf154 = aten.convolution_backward(buf152, mul_169, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf152
        del mul_169
        del primals_95
        buf155 = buf154[0]
        buf156 = buf154[1]
        del buf154
        buf157 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf157, buf58, mul_340, buf155, mul_379, 2097152, grid=grid(2097152), stream=stream0)
        del mul_340
        del mul_379
        buf158 = buf110; del buf110  # reuse
        buf159 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf165 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf160 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf166 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_45.run(buf157, convolution_25, unsqueeze_246, convolution_24, unsqueeze_258, squeeze_61, squeeze_58, buf158, buf159, buf165, buf160, buf166, 1024, 2048, grid=grid(1024), stream=stream0)
        buf161 = buf58; del buf58  # reuse
        buf167 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_46.run(buf157, convolution_25, unsqueeze_246, buf159, squeeze_61, buf158, primals_41, convolution_24, unsqueeze_258, buf165, squeeze_58, primals_39, buf161, buf167, 2097152, grid=grid(2097152), stream=stream0)
        del buf157
        del buf159
        del convolution_24
        del convolution_25
        del primals_39
        del primals_41
        del squeeze_58
        del squeeze_61
        del unsqueeze_246
        del unsqueeze_258
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf162 = aten.convolution_backward(buf161, mul_137, primals_94, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf161
        del primals_94
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf168 = aten.convolution_backward(buf167, mul_154, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf167
        del mul_154
        del primals_93
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = reinterpret_tensor(buf60, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf60  # reuse
        buf172 = reinterpret_tensor(buf171, (8, 1, 256), (256, 256, 1), 0); del buf171  # reuse
        # Source Nodes: [sigmoid_4, x_129], Original ATen: [aten.convolution_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_47.run(buf172, buf169, add_98, convolution_23, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
        buf173 = aten.convolution_backward(buf172, view_8, primals_92, [0], [1], [2], [1], False, [0], 1, [True, True, False])
        del buf172
        del primals_92
        del view_8
        buf174 = buf173[0]
        buf175 = buf173[1]
        del buf173
        buf176 = buf151; del buf151  # reuse
        buf177 = empty((256, ), device='cuda', dtype=torch.float32)
        buf179 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_48.run(buf169, convolution_23, buf174, add_98, convolution_22, unsqueeze_272, squeeze_55, buf176, buf177, buf179, 256, 2048, grid=grid(256), stream=stream0)
        buf178 = buf169; del buf169  # reuse
        buf180 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_49.run(buf180, convolution_23, buf174, add_98, convolution_22, unsqueeze_272, buf177, squeeze_55, buf176, primals_37, 524288, grid=grid(524288), stream=stream0)
        del add_98
        del buf174
        del convolution_22
        del convolution_23
        del primals_37
        del squeeze_55
        del unsqueeze_272
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf181 = aten.convolution_backward(buf180, mul_145, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del buf180
        del mul_145
        del primals_91
        buf182 = buf181[0]
        buf183 = buf181[1]
        del buf181
        buf184 = buf177; del buf177  # reuse
        buf185 = empty((256, ), device='cuda', dtype=torch.float32)
        buf186 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_50.run(buf182, mul_416, convolution_21, unsqueeze_284, squeeze_52, buf184, buf185, buf186, 256, 8192, grid=grid(256), stream=stream0)
        buf187 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_51.run(buf187, mul_416, convolution_21, unsqueeze_284, buf185, squeeze_52, buf184, primals_35, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_21
        del mul_416
        del primals_35
        del squeeze_52
        del unsqueeze_284
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf188 = aten.convolution_backward(buf187, mul_137, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf187
        del mul_137
        del primals_90
        buf189 = buf188[0]
        buf190 = buf188[1]
        del buf188
        buf191 = buf103; del buf103  # reuse
        buf192 = empty((512, ), device='cuda', dtype=torch.float32)
        buf194 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_52.run(buf163, buf189, mul_428, convolution_20, unsqueeze_296, squeeze_49, buf191, buf192, buf194, 512, 8192, grid=grid(512), stream=stream0)
        buf193 = empty((8, 512, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_53.run(buf163, buf189, mul_428, convolution_20, unsqueeze_296, buf192, squeeze_49, buf191, primals_33, buf193, 4194304, grid=grid(4194304), stream=stream0)
        del convolution_20
        del primals_33
        del squeeze_49
        del unsqueeze_296
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf195 = aten.convolution_backward(buf193, mul_129, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf193
        del mul_129
        del primals_89
        buf196 = buf195[0]
        buf197 = buf195[1]
        del buf195
        buf198 = reinterpret_tensor(buf165, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf165  # reuse
        buf199 = reinterpret_tensor(buf198, (8, 1, 128), (128, 128, 1), 0); del buf198  # reuse
        # Source Nodes: [sigmoid_3, x_106], Original ATen: [aten.convolution_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_54.run(buf199, buf196, add_82, convolution_19, 1024, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
        buf200 = aten.convolution_backward(buf199, view_6, primals_88, [0], [1], [2], [1], False, [0], 1, [True, True, False])
        del buf199
        del primals_88
        del view_6
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = empty((128, ), device='cuda', dtype=torch.float32)
        buf204 = empty((128, ), device='cuda', dtype=torch.float32)
        buf206 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55.run(buf196, convolution_19, buf201, add_82, convolution_18, unsqueeze_310, squeeze_46, buf203, buf204, buf206, 128, 8192, grid=grid(128), stream=stream0)
        buf205 = buf196; del buf196  # reuse
        buf207 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_56.run(buf207, convolution_19, buf201, add_82, convolution_18, unsqueeze_310, buf204, squeeze_46, buf203, primals_31, 1048576, grid=grid(1048576), stream=stream0)
        del add_82
        del convolution_18
        del convolution_19
        del primals_31
        del squeeze_46
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf208 = aten.convolution_backward(buf207, mul_120, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf207
        del mul_120
        del primals_87
        buf209 = buf208[0]
        buf210 = buf208[1]
        del buf208
        buf211 = buf204; del buf204  # reuse
        buf212 = empty((128, ), device='cuda', dtype=torch.float32)
        buf213 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_57.run(buf209, mul_456, convolution_17, unsqueeze_322, squeeze_43, buf211, buf212, buf213, 128, 8192, grid=grid(128), stream=stream0)
        buf214 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_58.run(buf214, mul_456, convolution_17, unsqueeze_322, buf212, squeeze_43, buf211, primals_29, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_17
        del mul_456
        del primals_29
        del squeeze_43
        del unsqueeze_322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf215 = aten.convolution_backward(buf214, mul_112, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf214
        del mul_112
        del primals_86
        buf216 = buf215[0]
        buf217 = buf215[1]
        del buf215
        buf218 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_59.run(buf218, buf189, mul_428, buf216, mul_468, 4194304, grid=grid(4194304), stream=stream0)
        del mul_428
        del mul_468
        buf219 = buf192; del buf192  # reuse
        buf220 = empty((512, ), device='cuda', dtype=torch.float32)
        buf226 = empty((512, ), device='cuda', dtype=torch.float32)
        buf221 = empty((512, ), device='cuda', dtype=torch.float32)
        buf227 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_60.run(buf218, convolution_16, unsqueeze_334, convolution_15, unsqueeze_346, squeeze_40, squeeze_37, buf219, buf220, buf226, buf221, buf227, 512, 8192, grid=grid(512), stream=stream0)
        buf222 = buf216; del buf216  # reuse
        buf228 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_61.run(buf218, convolution_16, unsqueeze_334, buf220, squeeze_40, buf219, primals_27, convolution_15, unsqueeze_346, buf226, squeeze_37, primals_25, buf222, buf228, 4194304, grid=grid(4194304), stream=stream0)
        del buf218
        del convolution_15
        del convolution_16
        del primals_25
        del primals_27
        del squeeze_37
        del squeeze_40
        del unsqueeze_334
        del unsqueeze_346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf223 = aten.convolution_backward(buf222, mul_80, primals_85, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf222
        del primals_85
        buf224 = buf223[0]
        buf225 = buf223[1]
        del buf223
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf229 = aten.convolution_backward(buf228, mul_97, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf228
        del mul_97
        del primals_84
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf232 = reinterpret_tensor(buf201, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf201  # reuse
        buf233 = reinterpret_tensor(buf232, (8, 1, 128), (128, 128, 1), 0); del buf232  # reuse
        # Source Nodes: [sigmoid_2, x_78], Original ATen: [aten.convolution_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_54.run(buf233, buf230, add_61, convolution_14, 1024, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
        buf234 = aten.convolution_backward(buf233, view_4, primals_83, [0], [1], [2], [1], False, [0], 1, [True, True, False])
        del primals_83
        del view_4
        buf235 = buf234[0]
        buf236 = buf234[1]
        del buf234
        buf237 = buf212; del buf212  # reuse
        buf238 = empty((128, ), device='cuda', dtype=torch.float32)
        buf240 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55.run(buf230, convolution_14, buf235, add_61, convolution_13, unsqueeze_360, squeeze_34, buf237, buf238, buf240, 128, 8192, grid=grid(128), stream=stream0)
        buf239 = buf230; del buf230  # reuse
        buf241 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_56.run(buf241, convolution_14, buf235, add_61, convolution_13, unsqueeze_360, buf238, squeeze_34, buf237, primals_23, 1048576, grid=grid(1048576), stream=stream0)
        del add_61
        del convolution_13
        del convolution_14
        del primals_23
        del squeeze_34
        del unsqueeze_360
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf242 = aten.convolution_backward(buf241, mul_88, primals_82, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf241
        del mul_88
        del primals_82
        buf243 = buf242[0]
        buf244 = buf242[1]
        del buf242
        buf245 = reinterpret_tensor(buf226, (128, 4), (1, 128), 0); del buf226  # reuse
        buf247 = reinterpret_tensor(buf220, (128, 4), (1, 128), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_62.run(buf243, mul_505, convolution_12, unsqueeze_372, buf245, buf247, 512, 8192, grid=grid(512), stream=stream0)
        buf246 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_63.run(buf245, buf246, 128, 4, grid=grid(128), stream=stream0)
        del buf245
        buf248 = empty((128, ), device='cuda', dtype=torch.float32)
        buf249 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_64.run(buf247, squeeze_31, buf248, buf249, 128, 4, grid=grid(128), stream=stream0)
        buf250 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_65.run(buf250, mul_505, convolution_12, unsqueeze_372, buf248, squeeze_31, buf246, primals_21, 4194304, grid=grid(4194304), stream=stream0)
        del buf248
        del convolution_12
        del mul_505
        del primals_21
        del squeeze_31
        del unsqueeze_372
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf251 = aten.convolution_backward(buf250, mul_80, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf250
        del mul_80
        del primals_81
        buf252 = buf251[0]
        buf253 = buf251[1]
        del buf251
        buf254 = buf185; del buf185  # reuse
        buf255 = empty((256, ), device='cuda', dtype=torch.float32)
        buf257 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_66.run(buf224, buf252, mul_517, convolution_11, unsqueeze_384, squeeze_28, buf254, buf255, buf257, 256, 32768, grid=grid(256), stream=stream0)
        buf256 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_67.run(buf224, buf252, mul_517, convolution_11, unsqueeze_384, buf255, squeeze_28, buf254, primals_19, buf256, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_11
        del primals_19
        del squeeze_28
        del unsqueeze_384
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf258 = aten.convolution_backward(buf256, mul_72, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf256
        del mul_72
        del primals_80
        buf259 = buf258[0]
        buf260 = buf258[1]
        del buf258
        buf261 = reinterpret_tensor(buf247, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf247  # reuse
        buf262 = reinterpret_tensor(buf261, (8, 1, 64), (64, 64, 1), 0); del buf261  # reuse
        # Source Nodes: [sigmoid_1, x_55], Original ATen: [aten.convolution_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_red_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_68.run(buf262, buf259, add_45, convolution_10, 512, 4096, grid=grid(512), stream=stream0)
        # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
        buf263 = aten.convolution_backward(buf262, view_2, primals_79, [0], [1], [1], [1], False, [0], 1, [True, True, False])
        del buf262
        del primals_79
        del view_2
        buf264 = buf263[0]
        buf265 = buf263[1]
        del buf263
        buf266 = reinterpret_tensor(buf255, (64, 4), (1, 64), 0); del buf255  # reuse
        buf268 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf259, convolution_10, buf264, add_45, convolution_9, unsqueeze_398, buf266, buf268, 256, 8192, grid=grid(256), stream=stream0)
        buf267 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf266, buf267, 64, 4, grid=grid(64), stream=stream0)
        buf269 = empty((64, ), device='cuda', dtype=torch.float32)
        buf271 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71.run(buf268, squeeze_25, buf269, buf271, 64, 4, grid=grid(64), stream=stream0)
        buf270 = buf259; del buf259  # reuse
        buf272 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_72.run(buf272, convolution_10, buf264, add_45, convolution_9, unsqueeze_398, buf269, squeeze_25, buf267, primals_17, 2097152, grid=grid(2097152), stream=stream0)
        del add_45
        del convolution_10
        del convolution_9
        del primals_17
        del squeeze_25
        del unsqueeze_398
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf273 = aten.convolution_backward(buf272, mul_63, primals_78, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del buf272
        del mul_63
        del primals_78
        buf274 = buf273[0]
        buf275 = buf273[1]
        del buf273
        buf276 = buf268; del buf268  # reuse
        buf278 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf274, mul_545, convolution_8, unsqueeze_410, buf276, buf278, 256, 8192, grid=grid(256), stream=stream0)
        buf277 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf276, buf277, 64, 4, grid=grid(64), stream=stream0)
        buf279 = empty((64, ), device='cuda', dtype=torch.float32)
        buf280 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71.run(buf278, squeeze_22, buf279, buf280, 64, 4, grid=grid(64), stream=stream0)
        buf281 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74.run(buf281, mul_545, convolution_8, unsqueeze_410, buf279, squeeze_22, buf277, primals_15, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_8
        del mul_545
        del primals_15
        del squeeze_22
        del unsqueeze_410
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf282 = aten.convolution_backward(buf281, mul_55, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf281
        del mul_55
        del primals_77
        buf283 = buf282[0]
        buf284 = buf282[1]
        del buf282
        buf285 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_75.run(buf285, buf252, mul_517, buf283, mul_557, 8388608, grid=grid(8388608), stream=stream0)
        del mul_517
        del mul_557
        buf286 = reinterpret_tensor(buf278, (256, ), (1, ), 0); del buf278  # reuse
        buf287 = reinterpret_tensor(buf276, (256, ), (1, ), 0); del buf276  # reuse
        buf293 = empty((256, ), device='cuda', dtype=torch.float32)
        buf288 = empty((256, ), device='cuda', dtype=torch.float32)
        buf294 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_76.run(buf285, convolution_7, unsqueeze_422, convolution_6, unsqueeze_434, squeeze_19, squeeze_16, buf286, buf287, buf293, buf288, buf294, 256, 32768, grid=grid(256), stream=stream0)
        buf289 = buf283; del buf283  # reuse
        buf295 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_77.run(buf285, convolution_7, unsqueeze_422, buf287, squeeze_19, buf286, primals_13, convolution_6, unsqueeze_434, buf293, squeeze_16, primals_11, buf289, buf295, 8388608, grid=grid(8388608), stream=stream0)
        del buf285
        del convolution_6
        del convolution_7
        del primals_11
        del primals_13
        del squeeze_16
        del squeeze_19
        del unsqueeze_422
        del unsqueeze_434
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf290 = aten.convolution_backward(buf289, getitem_6, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf289
        del primals_76
        buf291 = buf290[0]
        buf292 = buf290[1]
        del buf290
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf296 = aten.convolution_backward(buf295, mul_40, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_40
        del primals_75
        buf297 = buf296[0]
        buf298 = buf296[1]
        del buf296
        buf299 = reinterpret_tensor(buf264, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf264  # reuse
        buf300 = reinterpret_tensor(buf299, (8, 1, 64), (64, 64, 1), 0); del buf299  # reuse
        # Source Nodes: [sigmoid, x_27], Original ATen: [aten.convolution_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_red_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_68.run(buf300, buf297, add_24, convolution_5, 512, 4096, grid=grid(512), stream=stream0)
        # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
        buf301 = aten.convolution_backward(buf300, view, primals_74, [0], [1], [1], [1], False, [0], 1, [True, True, False])
        del primals_74
        del view
        buf302 = buf301[0]
        buf303 = buf301[1]
        del buf301
        buf304 = reinterpret_tensor(buf293, (64, 4), (1, 64), 0); del buf293  # reuse
        buf306 = reinterpret_tensor(buf287, (64, 4), (1, 64), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69.run(buf297, convolution_5, buf302, add_24, convolution_4, unsqueeze_448, buf304, buf306, 256, 8192, grid=grid(256), stream=stream0)
        buf305 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf304, buf305, 64, 4, grid=grid(64), stream=stream0)
        buf307 = empty((64, ), device='cuda', dtype=torch.float32)
        buf309 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71.run(buf306, squeeze_13, buf307, buf309, 64, 4, grid=grid(64), stream=stream0)
        buf308 = buf297; del buf297  # reuse
        buf310 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_72.run(buf310, convolution_5, buf302, add_24, convolution_4, unsqueeze_448, buf307, squeeze_13, buf305, primals_9, 2097152, grid=grid(2097152), stream=stream0)
        del add_24
        del convolution_4
        del convolution_5
        del primals_9
        del squeeze_13
        del unsqueeze_448
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf311 = aten.convolution_backward(buf310, mul_31, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
        del buf310
        del mul_31
        del primals_73
        buf312 = buf311[0]
        buf313 = buf311[1]
        del buf311
        buf314 = buf306; del buf306  # reuse
        buf316 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_73.run(buf312, mul_594, convolution_3, unsqueeze_460, buf314, buf316, 256, 8192, grid=grid(256), stream=stream0)
        buf315 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_70.run(buf314, buf315, 64, 4, grid=grid(64), stream=stream0)
        del buf314
        buf317 = empty((64, ), device='cuda', dtype=torch.float32)
        buf318 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71.run(buf316, squeeze_10, buf317, buf318, 64, 4, grid=grid(64), stream=stream0)
        del buf316
        buf319 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_74.run(buf319, mul_594, convolution_3, unsqueeze_460, buf317, squeeze_10, buf315, primals_7, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_3
        del mul_594
        del primals_7
        del squeeze_10
        del unsqueeze_460
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf320 = aten.convolution_backward(buf319, getitem_6, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf319
        del getitem_6
        del primals_72
        buf321 = buf320[0]
        buf322 = buf320[1]
        del buf320
        buf323 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_78.run(buf323, buf321, 2097152, grid=grid(2097152), stream=stream0)
        del buf321
        buf324 = reinterpret_tensor(buf295, (8, 64, 128, 128), (1048576, 16384, 128, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        triton_poi_fused_add_max_pool2d_with_indices_backward_79.run(getitem_7, buf323, buf324, 8388608, grid=grid(8388608), stream=stream0)
        del buf323
        del getitem_7
        buf325 = reinterpret_tensor(buf235, (64, 16), (16, 1), 0); del buf235  # reuse
        buf327 = reinterpret_tensor(buf233, (64, 16), (16, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_80.run(buf324, mul_606, convolution_2, unsqueeze_472, buf325, buf327, 1024, 8192, grid=grid(1024), stream=stream0)
        buf326 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_81.run(buf325, buf326, 64, 16, grid=grid(64), stream=stream0)
        del buf325
        buf328 = empty((64, ), device='cuda', dtype=torch.float32)
        buf329 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_82.run(buf327, squeeze_7, buf328, buf329, 64, 16, grid=grid(64), stream=stream0)
        del buf327
        buf330 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_83.run(buf330, mul_606, convolution_2, unsqueeze_472, buf328, squeeze_7, buf326, primals_5, 8388608, grid=grid(8388608), stream=stream0)
        del buf328
        del convolution_2
        del mul_606
        del primals_5
        del squeeze_7
        del unsqueeze_472
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf331 = aten.convolution_backward(buf330, mul_15, primals_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf330
        del mul_15
        del primals_71
        buf332 = buf331[0]
        buf333 = buf331[1]
        del buf331
        buf334 = reinterpret_tensor(buf302, (32, 16), (16, 1), 0); del buf302  # reuse
        buf336 = reinterpret_tensor(buf300, (32, 16), (16, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_84.run(buf332, mul_618, convolution_1, unsqueeze_484, buf334, buf336, 512, 8192, grid=grid(512), stream=stream0)
        buf335 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_85.run(buf334, buf335, 32, 16, grid=grid(32), stream=stream0)
        del buf334
        buf337 = empty((32, ), device='cuda', dtype=torch.float32)
        buf338 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_86.run(buf336, squeeze_4, buf337, buf338, 32, 16, grid=grid(32), stream=stream0)
        del buf336
        buf339 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_87.run(buf339, mul_618, convolution_1, unsqueeze_484, buf337, squeeze_4, buf335, primals_3, 4194304, grid=grid(4194304), stream=stream0)
        del buf337
        del convolution_1
        del mul_618
        del primals_3
        del squeeze_4
        del unsqueeze_484
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf340 = aten.convolution_backward(buf339, mul_7, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf339
        del mul_7
        del primals_70
        buf341 = buf340[0]
        buf342 = buf340[1]
        del buf340
        buf343 = empty((24, 16), device='cuda', dtype=torch.float32)
        buf345 = empty((24, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_88.run(buf341, mul_630, convolution, unsqueeze_496, buf343, buf345, 384, 8192, grid=grid(384), stream=stream0)
        buf344 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_89.run(buf343, buf344, 24, 16, grid=grid(24), stream=stream0)
        del buf343
        buf346 = empty((24, ), device='cuda', dtype=torch.float32)
        buf347 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_90.run(buf345, squeeze_1, buf346, buf347, 24, 16, grid=grid(24), stream=stream0)
        del buf345
        buf348 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_91.run(buf348, mul_630, convolution, unsqueeze_496, buf346, squeeze_1, buf344, primals_1, 3145728, grid=grid(3145728), stream=stream0)
        del buf346
        del convolution
        del mul_630
        del primals_1
        del squeeze_1
        del unsqueeze_496
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf349 = aten.convolution_backward(buf348, primals_203, primals_69, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf348
        del primals_203
        del primals_69
        buf350 = buf349[1]
        return (buf347, buf344, buf338, buf335, buf329, buf326, buf318, buf315, buf309, buf305, buf294, buf286, buf288, buf286, buf280, buf277, buf271, buf267, buf257, buf254, buf249, buf246, buf240, buf237, buf227, buf219, buf221, buf219, buf213, buf211, buf206, buf203, buf194, buf191, buf186, buf184, buf179, buf176, buf166, buf158, buf160, buf158, buf153, buf150, reinterpret_tensor(buf129, (23, 16), (16, 1), 0), reinterpret_tensor(buf125, (23, 16), (16, 1), 0), buf118, buf116, buf112, buf109, buf105, buf102, reinterpret_tensor(buf80, (23, 16), (16, 1), 0), reinterpret_tensor(buf76, (23, 16), (16, 1), 0), buf69, buf67, buf62, buf52, buf55, buf52, buf48, buf45, reinterpret_tensor(buf23, (23, 16), (16, 1), 0), reinterpret_tensor(buf19, (23, 16), (16, 1), 0), buf12, buf10, buf5, buf3, buf350, buf342, buf333, buf322, buf313, buf303, buf298, buf292, buf284, buf275, buf265, buf260, buf253, buf244, buf236, buf231, buf225, buf217, buf210, buf202, buf197, buf190, buf183, buf175, buf170, buf164, buf156, buf149, buf145, buf115, buf108, buf101, buf97, buf66, buf59, buf51, buf44, buf40, buf9, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.int64)
    convolution_3 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_31 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_24 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_63 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((8, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_61 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_112 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_82 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    mul_129 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_137 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_145 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_98 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    mul_154 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_169 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_177 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((16384, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((16384, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_186 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_194 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_202 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((4096, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    view_48 = rand_strided((4096, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_211 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_226 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_234 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((4096, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((4096, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_243 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_51 = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_34 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_253 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_126 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_265 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    sub_40 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    permute_42 = rand_strided((64, 144, 64), (9216, 1, 144), device='cuda:0', dtype=torch.float32)
    permute_43 = rand_strided((64, 64, 144), (9216, 144, 1), device='cuda:0', dtype=torch.float32)
    alias_8 = rand_strided((64, 1, 64, 144), (9216, 9216, 144, 1), device='cuda:0', dtype=torch.float32)
    permute_47 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_55 = rand_strided((64, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_56 = rand_strided((64, 144, 16), (2304, 1, 144), device='cuda:0', dtype=torch.float32)
    mul_280 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_150 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_292 = rand_strided((8, 2048, 8, 8), (131072, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_174 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_313 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    sub_60 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((256, 144, 16), (2304, 1, 144), device='cuda:0', dtype=torch.float32)
    permute_69 = rand_strided((256, 64, 144), (9216, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_9 = rand_strided((64, 4, 16, 144), (9216, 2304, 144, 1), device='cuda:0', dtype=torch.float32)
    permute_73 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_79 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_81 = rand_strided((256, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_82 = rand_strided((256, 144, 16), (2304, 1, 144), device='cuda:0', dtype=torch.float32)
    mul_328 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_340 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_352 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    sub_76 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_94 = rand_strided((256, 144, 64), (9216, 1, 144), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((256, 32, 144), (4608, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_10 = rand_strided((64, 4, 64, 144), (36864, 9216, 144, 1), device='cuda:0', dtype=torch.float32)
    permute_99 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    permute_107 = rand_strided((256, 16, 64), (1024, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_108 = rand_strided((256, 144, 16), (2304, 1, 144), device='cuda:0', dtype=torch.float32)
    mul_367 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_379 = rand_strided((8, 1024, 16, 16), (262144, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_272 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_416 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_284 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_428 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_296 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_456 = rand_strided((8, 128, 32, 32), (131072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_468 = rand_strided((8, 512, 32, 32), (524288, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_360 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_505 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_372 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_517 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_384 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_545 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_557 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_448 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_594 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_460 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_606 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_472 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_618 = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_484 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_630 = rand_strided((8, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_496 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_203, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem_6, getitem_7, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, view_17, view_23, squeeze_67, mul_186, convolution_29, squeeze_70, mul_194, convolution_30, squeeze_73, mul_202, view_42, view_48, squeeze_76, mul_211, convolution_33, squeeze_79, convolution_34, squeeze_82, mul_226, convolution_35, squeeze_85, mul_234, view_67, view_73, squeeze_88, mul_243, convolution_38, squeeze_91, clone_51, permute_34, mul_253, unsqueeze_126, mul_265, sub_40, permute_42, permute_43, alias_8, permute_47, permute_53, permute_55, permute_56, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, sub_60, permute_68, permute_69, alias_9, permute_73, permute_79, permute_81, permute_82, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, sub_76, permute_94, permute_95, alias_10, permute_99, permute_105, permute_107, permute_108, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('eca_halonext26ts', benchmark_compiled_module)
