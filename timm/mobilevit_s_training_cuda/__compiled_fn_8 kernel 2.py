
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


# kernel path: /tmp/torchinductor_youkaichao/fe/cfe3fawfcwkpzct3jrnenygbsycbzlkvdrgqs26xlkybtp5oyour.py
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_native_batch_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 640
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
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgzis5wsrnhbybjuptul5momixukwjrtvnjbo2vnvectpn2iay6.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 64)
    x4 = xindex
    x1 = (xindex // 64) % 640
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


# kernel path: /tmp/torchinductor_youkaichao/4n/c4ny3urp4kpuy6zklnd2aysoxxvznmdzpwi6co5b5smw5r7lrx74.py
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
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (10240*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (10240*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0) + (10240*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ce/cceu3oldbhc35xihgzdi6vvdurt5vtgwilfsv4ip7d6esbup2cgv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 160
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
    tmp7 = 0.001953125
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


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vcnrcilhclltabxt3ccedjbcrhgtkwtlawipd4l2xqzgmzsky6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (10240 + r1 + (64*x0) + (20480*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (10240*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0) + (10240*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7pkdqt5lo65t6uzzriaydpkxhbz6dhkzubuyfsaqokhtovfmdi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10240)
    x3 = xindex % 10240
    x4 = xindex
    x1 = (xindex // 64) % 160
    tmp0 = tl.load(in_ptr0 + (10240 + x3 + (20480*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp3 = tl.load(in_ptr2 + (x4), None)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.001953125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x4), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3mdaib3sebfe35l7t5byyhqkh77q5gtfvivudan37vvvh2qc5eq.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r3) + (7680*x2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 7680)) + ((x0 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (120*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwme6rxz5qu2oniiyl3lx7wv4eufogs23q7lbry4ef22m4qzralg.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + (32*x0) + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ce/ccedx5z57q3gf7jkfnz6ios3vecedw5ipnbygzu7oe2twedcscqo.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 16
    x2 = (xindex // 32)
    x5 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x2 % 4)) // 4) % 4)) + (8*((x2 % 4) // 2)) + (16*((((4*x1) + (64*r3) + (7680*x0) + (15360*(x2 // 4)) + (x2 % 4)) // 16) % 7680)) + ((x2 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (120*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (120*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/22/c22ey2ut2cgw4mhqgvectipfkgcrodiyzt75e7r2dpuv6tyn6zmy.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g6/cg67iu46hi2rakxt5qu7mkpsb55z4drovtzkk2tl5xaj2zj6xfmw.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_poi_fused_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 3840
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 240)
    y0 = yindex
    x1 = xindex % 240
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((2*((((4*x2) + (y0 % 4)) // 4) % 4)) + (8*((y0 % 4) // 2)) + (16*((((4*x2) + (64*x1) + (15360*(y0 // 4)) + (y0 % 4)) // 16) % 7680)) + ((y0 % 4) % 2)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x3 + (3840*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 240.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x3 + (3840*y0)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xu7gsassnddiax7aahyilgzebigkbxtqlpux5eorvkmc6czc76.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*(r2 % 16)) + ((r2 // 16) % 4)) // 4) % 4)) + (8*(((r2 // 16) % 4) // 2)) + (16*((((4*(r2 % 16)) + (64*x0) + (15360*(r2 // 64)) + (30720*x1) + ((r2 // 16) % 4)) // 16) % 7680)) + (((r2 // 16) % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tzkz3j6fj5g7pau57ehjjgsxbjynu3serktwkp7pl2cyl5vlgm.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazdol2v2ahdykpqmr73cta3omuamwxiyvfoezgdon7u5lta4fhp.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 240
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 32
    r2 = (rindex // 32)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*r2) + (r1 % 4)) // 4) % 4)) + (8*((r1 % 4) // 2)) + (16*((((4*r2) + (64*x0) + (15360*(r1 // 4)) + (r1 % 4)) // 16) % 7680)) + ((r1 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwqryj2rcks3nj5fxzhur2hqakogzcycjnsoin2bumsg4u5iienq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3x/c3xemhrroa37v45n7tjfzlah2umepv5h6iro7gp6aazmzb6pmlbc.py
# Source Nodes: [x_325], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
# x_325 => sigmoid_30
triton_poi_fused_add_fill_mul_silu_sub_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_silu_sub_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cilidxj5o4euth57exm3a7t2tbj5txymtjjaixyx227bvnbj2pna.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (61440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dra3smi5ukkdtuo7ewwqcvav4hfvdtvzujhz6fts4ia7u7vndn.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yl/cylpbo6oerhsktvkduavgr73caktjnbsnll5erhhfzlcizl2ujwr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 240.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (240*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7u7ehobgu45vkfjewwiqxuzama2w5bewv64xm4nqe5e7bilktzw.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wv/cwvcotvgeogcr7reeq2emfiedpjmktdmz433agzijnebvewqxebb.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 368640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 3840)
    x6 = xindex
    x0 = xindex % 240
    x3 = (xindex // 122880)
    x7 = (xindex // 240) % 512
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 64, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-122880) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 96, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-245760) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (240*x3) + (720*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfvnqg2rvzexnusfpcr37nz4wwe655hvjyctvy23ugbc7wgtfvt.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2880
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 720
    x1 = (xindex // 720)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (720*r2) + (92160*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdo2w5viejsmravdnq2ts7vsur6b5m2ujbaxqoskzzfnf5zw4uca.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmn73b4cwkxd4qttif2ingfuxvpq7nfqphdqpzj46b5ajthoge3z.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcx7m5co6p7ooqns25s6lbv6cf6lhpwevi6jfqi7z6tp44axshb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 240
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
    tmp0 = tl.load(in_ptr0 + ((240*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (y0 % 2)) // 4) % 16)) + (3840*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (15360*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (15360*y2) + (y0 % 2)) // 15360) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (y0 % 2)) // 64) % 240)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((16*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (64*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (15360*y2) + (y0 % 2)) // 15360) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (y0 % 2)) // 4) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + ((240*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (y0 % 2)) // 4) % 16)) + (3840*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (15360*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (15360*y2) + (y0 % 2)) // 15360) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (y0 % 2)) // 64) % 240)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (y0 % 2)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + ((16*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (64*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (15360*y2) + (y0 % 2)) // 15360) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (y0 % 2)) // 4) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + ((240*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (y0 % 2)) // 4) % 16)) + (3840*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (15360*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (15360*y2) + (y0 % 2)) // 15360) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (y0 % 2)) // 64) % 240)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + ((16*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (64*((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (64*x3) + (15360*y2) + (y0 % 2)) // 15360) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (16*((y0 + (8*y1)) // 16)) + (y0 % 2)) // 4) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 240.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tl.store(out_ptr0 + (y4 + (64*x3) + (15360*y2)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cby6gihsenvrhftulo6g3sbwm4erirze3ptp5jvlsficlbbu6nbh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (20480*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (10240*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0) + (10240*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czsven25oc63jugkp2mipqjavmq6jefaxjw7e243xspsl74jwiqu.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10240)
    x3 = xindex % 10240
    x4 = xindex
    x1 = (xindex // 64) % 160
    tmp0 = tl.load(in_ptr0 + (x3 + (20480*x2)), None)
    tmp1 = tl.load(in_out_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + (x4), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.001953125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x4), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckegaeigkb2gh345vfbgep3ic6v4hinnp5amgskwn6e4rxdaxcx.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_28', 'mutated_arg_names': []}
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
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ns/cnsut5hwl533fmefrnukn5p4ka23gzn26s36hmyfpk2ujikp3ufs.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_29', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp7 = 0.001953125
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


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwe4kloitlxfzvjxirsunquw5ifuqdqui72pnvehhd5x47zzxz4.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_30', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/om/comy64wacpqzbfrbdvkxjryrvsufw5shxt3mb7w6utftefoidmqh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_31', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp7 = 0.00048828125
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


# kernel path: /tmp/torchinductor_youkaichao/ld/cld7nyj3ilpoa2lecicx2jpmk3rmv22peslworzrwmtxun2kjzsk.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
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
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zzxx2umsacpgonzfqsqq6gkyxu33oi6dwv2hue5ynmg7gxpnnv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 128
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
    tmp7 = 0.00048828125
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


# kernel path: /tmp/torchinductor_youkaichao/pw/cpwrxsa5octwn5kvya62n3n2qgfg5ixt7mtxobtk2dboheuw26u7.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
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
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (32768 + r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/d6/cd6hgwr5kbgrc7j2q64xjy5f5oabpnpqtlewy3v4hszyr2g2o3lx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 32768)
    x3 = xindex % 32768
    x4 = xindex
    x1 = (xindex // 256) % 128
    tmp0 = tl.load(in_ptr0 + (32768 + x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp3 = tl.load(in_ptr2 + (x4), None)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00048828125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x4), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rqyayr3e2fx6mbas7k2fgrgq3ksqfwbq3huvjau55yfvckrmiq.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x2 = (xindex // 2048)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r3) + (24576*x2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 12288)) + ((x0 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hj/chjpcbtmczmrpwvx72zwxzzb4332gpfutzsv2hzbfvehvxp5eish.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + (32*x0) + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwywhiztdxzdagwspt3lkbc2hv3mr33f6xma3gw3qbnlgbevgti.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 64
    x2 = (xindex // 128)
    x5 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x2 % 4)) // 4) % 8)) + (16*((x2 % 4) // 2)) + (32*((((4*x1) + (256*r3) + (24576*x0) + (49152*(x2 // 4)) + (x2 % 4)) // 32) % 12288)) + ((x2 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (96*x5)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7qvojknr6n4g5oyozzz7h6hajqkmfga6twnlr2thzx7cawp3r4.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvf5giz2zulozc2nihpzjpjf6jmia6jvpxw65ej7q3g4kho5344w.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_poi_fused_native_layer_norm_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 12288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 192)
    y0 = yindex
    x1 = xindex % 192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((2*((((4*x2) + (y0 % 4)) // 4) % 8)) + (16*((y0 % 4) // 2)) + (32*((((4*x2) + (256*x1) + (49152*(y0 // 4)) + (y0 % 4)) // 32) % 12288)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x3 + (12288*y0)), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 192.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x3 + (12288*y0)), tmp12, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/consvcgx4ns33e24zufl2w4eksj2qnhckcwlw2ds32gac6chzfns.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*(r2 % 64)) + (((2*x0) + (r2 // 64)) % 4)) // 4) % 8)) + (16*((((2*x0) + (r2 // 64)) % 4) // 2)) + (32*((((4*(r2 % 64)) + (256*x1) + (49152*(((2*x0) + (r2 // 64)) // 4)) + (((2*x0) + (r2 // 64)) % 4)) // 32) % 12288)) + ((((2*x0) + (r2 // 64)) % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctsyz57mui7tppfs6o5d3dbtt5txjjlhy2edx3ip67xzmxwxiyw.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/bu/cbuqiu7bulft5foiwlfjk7max3jj5q43nfeenezzymiurk4wjbrb.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 32
        r2 = (rindex // 32)
        tmp0 = tl.load(in_ptr0 + ((2*((((4*r2) + (r1 % 4)) // 4) % 8)) + (16*((r1 % 4) // 2)) + (32*((((4*r2) + (256*x0) + (49152*(r1 // 4)) + (r1 % 4)) // 32) % 12288)) + ((r1 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rfcer2a6c2vehmbl6xbirhawh6mdtdtwfpgbu2ib5jaj535eyc.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgwgt6wjeqsitcoqh3cmpnu6ljss5wyumfgjpdrz4yxr7ln6dmb.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxnojikjjvbdj6do4klphvhjmpiktby3hfw7xm5rpgjxg2mgy3b.py
# Source Nodes: [x_241], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
# x_241 => sigmoid_22
triton_poi_fused_add_fill_mul_silu_sub_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_silu_sub_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/coziiwenmnyghrj53vlfjsgiphwjmeasznwl3qdwggfazrypiymd.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47hacro54lan7aaobjos367kvkfs7gqc5t7getfjwwou2vhz4re.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvu5vxcubf3cumnwzuurzdih2qnfw3gvey7y7v7emcf5becf3wne.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_49', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (192*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 192.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dyds54n2g75pz4arfiuez5yomvwrmvenidl6wctsjqoi3r2ird.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cunyyxfwwd3yxghif62rfftzt5dt6eapedjjkfeobc7ailfpg6qy.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 12288)
    x6 = xindex
    x0 = xindex % 192
    x3 = (xindex // 393216)
    x7 = (xindex // 192) % 2048
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 64, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-393216) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 96, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-786432) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (192*x3) + (576*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kr/ckrvtwcu55bhwpxjyvinhmhzqdojcvjxitabfqg7hoe4fdsqjj75.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (73728*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wz/cwz7b7v5jukj4jihyzhgl6bo3aaovh2pzbh4xikju36u6rojrkt3.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgkuwqulunfen37y2xr2xhcmap23uejsi7fuggtmphmx2747d6j.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (192*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpf24rq6uhkss4abjdhvpr72ofb4nwjmvclstqhgsfvwm5pphu3v.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + ((192*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (y0 % 2)) // 4) % 64)) + (12288*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (49152*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (49152*y2) + (y0 % 2)) // 49152) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (y0 % 2)) // 256) % 192)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((64*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (256*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (49152*y2) + (y0 % 2)) // 49152) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (y0 % 2)) // 4) % 64)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + ((192*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (y0 % 2)) // 4) % 64)) + (12288*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (49152*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (49152*y2) + (y0 % 2)) // 49152) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (y0 % 2)) // 256) % 192)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (y0 % 2)) // 256) % 192), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + ((64*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (256*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (49152*y2) + (y0 % 2)) // 49152) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (y0 % 2)) // 4) % 64)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + ((192*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (y0 % 2)) // 4) % 64)) + (12288*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (49152*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (49152*y2) + (y0 % 2)) // 49152) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (y0 % 2)) // 256) % 192)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + ((64*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (256*((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (256*x3) + (49152*y2) + (y0 % 2)) // 49152) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (32*((y0 + (16*y1)) // 32)) + (y0 % 2)) // 4) % 64)), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 192.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tl.store(out_ptr0 + (y4 + (256*x3) + (49152*y2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ng/cngumepmq7d2wk7jlx4it7nzhqgelfvuhhbpfwem3bp7n3vpcpvn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
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
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/3z/c3z7vkv3cohudbm4rjia5sjwb2bpxgm23sjxy7fklbanjkbsknge.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 32768)
    x3 = xindex % 32768
    x4 = xindex
    x1 = (xindex // 256) % 128
    tmp0 = tl.load(in_ptr0 + (x3 + (65536*x2)), None)
    tmp1 = tl.load(in_out_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + (x4), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00048828125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x4), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tb/ctb7r6kfxx2sd4emd2pvjsl45pra7mhxbg6dxbn74iq7k3gplgzy.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 2048
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
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (256*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5grrttbdynyifmbgg32fy7fraiopfldzmlk5qdh26n5ox44pkq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 384
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
    tmp7 = 0.00048828125
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


# kernel path: /tmp/torchinductor_youkaichao/js/cjs6a3gihdnxbhz7e7vn4ez75zfvlwgoeu3shkovg3kv7fnqetsy.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (393216*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (393216*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0) + (393216*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgcexev66smnmourrhrtmuiqafz2mkh4c7hpf7dpctvdhdkwbpz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 384
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


# kernel path: /tmp/torchinductor_youkaichao/3v/c3va3yxskfydopzqleooozxkatkaz4mnc3e35h2btmpecqmxtypn.py
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
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnkxsn4lzkru5fhmylvifmsyfpx7ohrz4qo7h3xjvxlftlad2tn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_63', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 96
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


# kernel path: /tmp/torchinductor_youkaichao/lx/clxj3owsuc3cywkrowp3fj7i5addoo7r27qk4rjrgbpbzcqbn2yl.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (98304 + r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4e/c4eymhkjhb3t7z4stvuklnm6zc6lopsjcgm7dfrhpvvt6unolpab.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 98304)
    x3 = xindex % 98304
    x4 = xindex
    x1 = (xindex // 1024) % 96
    tmp0 = tl.load(in_ptr0 + (98304 + x3 + (196608*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp3 = tl.load(in_ptr2 + (x4), None)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckm2s7bajmv2xe2jtkmlv56sfcez4rt5yo4yytbdhlfmgqrhmhhi.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    x4 = xindex % 256
    x5 = (xindex // 256)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 16)) + (32*((x0 % 4) // 2)) + (64*((((4*x1) + (1024*r2) + (147456*(x0 // 4)) + (x0 % 4)) // 64) % 18432)) + ((x0 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr0 + ((2*((((4*x4) + (x5 % 4)) // 4) % 16)) + (32*((x5 % 4) // 2)) + (64*((((4*x4) + (1024*r2) + (147456*(x5 // 4)) + (x5 % 4)) // 64) % 18432)) + ((x5 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp7 = tmp6 * tmp1
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crzd5cyzxqngwpxqfzc5hdn4hmg52ubftninsuodrcxgk7vvcbvz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_poi_fused_native_layer_norm_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 36864
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 144)
    y0 = yindex
    x1 = xindex % 144
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y0)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((2*((((4*x2) + (y0 % 4)) // 4) % 16)) + (32*((y0 % 4) // 2)) + (64*((((4*x2) + (1024*x1) + (147456*(y0 // 4)) + (y0 % 4)) // 64) % 18432)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (32*x2)), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x3 + (36864*y0)), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2 + (256*y0)), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = 144.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x3 + (36864*y0)), tmp12, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmyjmfnggkj6wo67vny2mu7jjxn4nlj7dretjiv3gbkb4ds2n4yf.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*((r2 + (128*x0)) % 256)) + (((r2 + (128*x0)) // 256) % 4)) // 4) % 16)) + (32*((((r2 + (128*x0)) // 256) % 4) // 2)) + (64*((((4*((r2 + (128*x0)) % 256)) + (1024*x1) + (147456*((r2 + (128*x0)) // 1024)) + (((r2 + (128*x0)) // 256) % 4)) // 64) % 18432)) + ((((r2 + (128*x0)) // 256) % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpsj7pkvilwk4up3iuf7tad2hvjyuj4gbmsris7hnpjm5gzx3kgv.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6r/c6rqraybcqsoel6aylllqaqjmgik2gqwebftieb4cfqljtrfo5qk.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 32
        r2 = (rindex // 32)
        tmp0 = tl.load(in_ptr0 + ((2*((((4*r2) + (r1 % 4)) // 4) % 16)) + (32*((r1 % 4) // 2)) + (64*((((4*r2) + (1024*x0) + (147456*(r1 // 4)) + (r1 % 4)) // 64) % 18432)) + ((r1 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/24/c24ucak3bgpcqlxyvsstr7hv2qfi3e63w5ofliaqbfo6zebozxkg.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdl2mdra5efzxo4s3dvckv6dyrcftvtl266zfbblgerbehkzl5g.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7ynqrlelce7tc6qa5zaolvcbtxfs3clixoltivenkxvt4xy3zqs.py
# Source Nodes: [x_145], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
# x_145 => sigmoid_13
triton_poi_fused_add_fill_mul_silu_sub_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_silu_sub_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhkweoigleospd5xbkrmxagyeqc3les7huwlk5d2vj6q4e5qly3.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 288
    x1 = (xindex // 288)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (288*r2) + (36864*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpedeypzhe4kofund5iathngbojrigcz2kkq4ptyak4ko2ajtoia.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (288*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ub/cub3vdrxvcac6a3ru74kyvhmko3siaydnhctu5o7d2e3emde7knk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_76', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (144*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 144.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54fgj2imohcvzzigvutlu4j7l2couimykzmhpqk3vbyzg62jkd6.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tcuuayg54f5mcegyojxkj7rw6fnqisqwybdekrhd67ilcbwph6.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3538944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 36864)
    x6 = xindex
    x0 = xindex % 144
    x3 = (xindex // 1179648)
    x7 = (xindex // 144) % 8192
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 64, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-1179648) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 96, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-2359296) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (144*x3) + (432*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxl4siiu4uj3ehivjefl5jx4hqgqxpgulzteiqnh476g52idfnr.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 27648
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 432
    x1 = (xindex // 432)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (432*r2) + (55296*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7f2p6a5thjm6pfjckc77ftbdfoevzeuiiwzwbdrrl7dgoqfuawu.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (432*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccngcq6cmgez6x6cbwbblawrw2u2zinz4a3qqdejmrxoc5p2xemp.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqnoopfx2sgndqc2deqy4wmhtkyiqpugnurjkvimiqp6xk3f2ur.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32) % 32
    y2 = (yindex // 1024)
    y4 = yindex % 1024
    tmp0 = tl.load(in_ptr0 + ((144*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)) + (36864*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (147456*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (y0 % 2)) // 1024) % 144)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((256*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (1024*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + ((144*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)) + (36864*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (147456*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (y0 % 2)) // 1024) % 144)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (y0 % 2)) // 1024) % 144), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + ((256*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (1024*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + ((144*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)) + (36864*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (147456*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (y0 % 2)) // 1024) % 144)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + ((256*(((2*(y1 % 2)) + (y0 % 2)) % 4)) + (1024*((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (1024*x3) + (147456*y2) + (y0 % 2)) // 147456) % 8)) + ((((2*(y1 % 2)) + (4*(y0 // 2)) + (64*((y0 + (32*y1)) // 64)) + (y0 % 2)) // 4) % 256)), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 144.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tl.store(out_ptr0 + (y4 + (1024*x3) + (147456*y2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdeiadhqgrxetqi6nsmcqjy5fidlpsq3dscmnxpcsc7bsupaewv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/zc/czcqv72wketeeo2any46n2gsc6po3oiqj2y2rmwthfse4qo6lurg.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_84', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 98304)
    x3 = xindex % 98304
    x4 = xindex
    x1 = (xindex // 1024) % 96
    tmp0 = tl.load(in_ptr0 + (x3 + (196608*x2)), None)
    tmp1 = tl.load(in_out_ptr0 + (x4), None)
    tmp3 = tl.load(in_ptr1 + (x4), None)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
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
    tl.store(in_out_ptr0 + (x4), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkhdym4nqr2xro3zi5tdql3qfyq5nrwn2ekqngeoha7maupnhs2.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_85', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wr/cwrecgvlrgvbhu77bk5nrev34m7yvcsuvayb75xuetvxkkqszxar.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_86', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ol/colc4u5u5vo4fucljqz4r3rfjx2py6sxa2lf5m7bopjgknntmehz.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
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
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqcnqxpiicieofsp72wyqj4bvudxyqvknngvbb6yimx7ylcotlb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_88', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
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


# kernel path: /tmp/torchinductor_youkaichao/5k/c5ksibeep7yj7ejzzyppaeejcsmxnnovjqpf5qeonqo265nnlm4h.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5o/c5ozulfdhzyx6npzrep7gdjzpiwnwvaufhc5w2npzbtbuicww3f2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_90', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/v7/cv7hq36uqkyimkph5yg24c5wa5pi3sgmkitgeievjtj6dfwlutfu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_91', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/po/cpo4wmks2ebz2p5ocnsmd52i2xjosnwaiorwejce3bai5yyuezgv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
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


# kernel path: /tmp/torchinductor_youkaichao/vk/cvk4yckyg5bypoz6jelhpwqwnq5iluqhhqceqzmmd6duw5nrsodx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_93', 'mutated_arg_names': []}
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
        tmp2 = tmp0 + tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/2g/c2gogzgxz4rnpztgh2avbejsbkdds64vajqdygt5yunnndtma3dd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
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
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcf726hmx2a7ffpz5sd5zbemhk26bzzjvkvu45farydguz66rxp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfpvi4ot27pujgtefapc6dips44vwshyz4oianyxs7nc7ltvysp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_96', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
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
    tmp4 = tmp2 + tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wthrlqqa4mea42v5yod3ebvlfvjo5ta4poj7i2wqywnyzazkcl.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_97', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hz/chz7xhn4tx7pvf5faxejpmofe6opmeqthgir7kyy5ghfuitdot6t.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_98', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yx/cyx3chw6edq76xbjteqdu6ucgfg47wqhc56ykn5vvfdapwgqiaft.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_99', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfx5d4qj7srzhcgl6beakd43n2l3cz7wy5h4nxecjgyejpqjxyj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_100', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/zy/czycpibsetlljkrn72lqvmfjetbvck5t66mrultxug3ajj4zd4as.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_101', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/dl/cdl5ig47gux6a3hm2mmyf4v5jvc4zya4oudj5mkrwfo35qzdyfkq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_102', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bs/cbswdja5idhempg5yvc3ey7rhysdjlgwgfnpm4hjwsvyvpuuftgs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzwxtu2mxnjxqpk4cr4ni2tkfnuivhyfqcyganzyo3expa3y3ox.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_104', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3a/c3a3awtceeuxcljnvrvs4iggtbhe62opigx3ygg6mldoe36qdzct.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_105', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jg/cjgtc7fzupvxxcdytzgscdylqrknsn4gq327lebkc7lo7tzwzmxb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_106', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 7.62939453125e-06
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uc/cucn3u5fqzkb4whl6nehk3wdw4vg7lwqdmxolv5ejyll4my5n6sc.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_107', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ef/cefoha2gzjvbjyojb4u42gsemirjefgshmwe5i5rdomoylfqftmv.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_108', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnjrt66voribbukmka2hs3sake7nunnl34j4a2yzfz6u2ssb2q3.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_109', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hw/chwdosjfuu4df67nd7wd23v6hwi34p76jcfxnqzfcdguhm7niwys.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_110', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvpavrclmtyur2fjep22a3pzxudq3f5vyu3hll2xevotehscjah.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (262144*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (262144*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (262144*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqyf7bzemx3aoifq63itt4ixyvtyss2r7awsjgt2ayshmmtyksi.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_112', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlxtvr75dsoaerhsdx7jeqlixwiwgo6re4ahzciel5cp7qmqcfs.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_113 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/sx/csxjnsi5vg4bpmjlmh6os5yvk35llwpg7vx2fttyazipyonaygqh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_114', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 16
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_89, primals_95, primals_101, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_152, primals_158, primals_164, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_211, primals_212, primals_213, primals_312, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, add_19, convolution_4, squeeze_13, mul_38, convolution_5, squeeze_16, mul_46, convolution_6, squeeze_19, add_34, convolution_7, squeeze_22, mul_61, convolution_8, squeeze_25, mul_69, convolution_9, squeeze_28, add_50, convolution_10, squeeze_31, mul_84, convolution_11, squeeze_34, mul_92, convolution_12, squeeze_37, add_66, convolution_13, squeeze_40, mul_107, convolution_14, squeeze_43, mul_115, convolution_15, squeeze_46, add_81, convolution_16, squeeze_49, mul_130, mul_131, view_3, getitem_36, getitem_37, getitem_38, getitem_40, getitem_41, getitem_42, view_7, mul_133, view_9, addmm_2, view_11, mul_136, view_13, getitem_47, getitem_48, getitem_49, getitem_51, getitem_52, getitem_53, view_17, mul_138, view_19, addmm_6, view_21, mul_141, view_25, convolution_18, squeeze_52, cat, convolution_19, squeeze_55, mul_158, convolution_20, squeeze_58, mul_166, convolution_21, squeeze_61, mul_174, convolution_22, squeeze_64, add_125, convolution_23, squeeze_67, mul_189, mul_190, view_29, getitem_72, getitem_73, getitem_74, getitem_76, getitem_77, getitem_78, view_33, mul_192, view_35, addmm_10, view_37, mul_195, view_39, getitem_83, getitem_84, getitem_85, getitem_87, getitem_88, getitem_89, view_43, mul_197, view_45, addmm_14, view_47, mul_200, view_49, getitem_94, getitem_95, getitem_96, getitem_98, getitem_99, getitem_100, view_53, mul_202, view_55, addmm_18, view_57, mul_205, view_59, getitem_105, getitem_106, getitem_107, getitem_109, getitem_110, getitem_111, view_63, mul_207, view_65, addmm_22, view_67, mul_210, view_71, convolution_25, squeeze_70, cat_1, convolution_26, squeeze_73, mul_227, convolution_27, squeeze_76, mul_235, convolution_28, squeeze_79, mul_243, convolution_29, squeeze_82, add_181, convolution_30, squeeze_85, mul_258, mul_259, view_75, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, view_79, mul_261, view_81, addmm_26, view_83, mul_264, view_85, getitem_141, getitem_142, getitem_143, getitem_145, getitem_146, getitem_147, view_89, mul_266, view_91, addmm_30, view_93, mul_269, view_95, getitem_152, getitem_153, getitem_154, getitem_156, getitem_157, getitem_158, view_99, mul_271, view_101, addmm_34, view_103, mul_274, view_107, convolution_32, squeeze_88, cat_2, convolution_33, squeeze_91, mul_291, convolution_34, squeeze_94, clone_64, permute_67, mul_301, unsqueeze_130, mul_313, unsqueeze_142, mul_325, unsqueeze_154, div_1, permute_76, permute_81, div_2, permute_85, alias_9, permute_91, div_3, permute_95, permute_100, div_4, permute_104, alias_10, permute_110, div_5, permute_114, permute_119, div_6, permute_123, alias_11, permute_129, div_7, mul_395, unsqueeze_166, unsqueeze_178, mul_416, unsqueeze_190, mul_428, unsqueeze_202, mul_440, unsqueeze_214, mul_452, unsqueeze_226, div_8, permute_142, permute_147, div_9, permute_151, alias_12, permute_157, div_10, permute_161, permute_166, div_11, permute_170, alias_13, permute_176, div_12, permute_180, permute_185, div_13, permute_189, alias_14, permute_195, div_14, permute_199, permute_204, div_15, permute_208, alias_15, permute_214, div_16, mul_539, unsqueeze_238, unsqueeze_250, mul_560, unsqueeze_262, mul_572, unsqueeze_274, mul_584, unsqueeze_286, mul_596, unsqueeze_298, div_17, permute_227, permute_232, div_18, permute_236, alias_16, permute_242, div_19, permute_246, permute_251, div_20, permute_255, alias_17, permute_261, div_21, mul_649, unsqueeze_310, unsqueeze_322, mul_670, unsqueeze_334, mul_682, unsqueeze_346, unsqueeze_358, mul_703, unsqueeze_370, mul_715, unsqueeze_382, unsqueeze_394, mul_736, unsqueeze_406, mul_748, unsqueeze_418, unsqueeze_430, mul_769, unsqueeze_442, mul_781, unsqueeze_454, unsqueeze_466, mul_802, unsqueeze_478, mul_814, unsqueeze_490, mul_826, unsqueeze_502, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_37, (96, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (160, ), (1, ))
    assert_size_stride(primals_57, (160, ), (1, ))
    assert_size_stride(primals_59, (160, ), (1, ))
    assert_size_stride(primals_61, (160, ), (1, ))
    assert_size_stride(primals_63, (640, ), (1, ))
    assert_size_stride(primals_65, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_66, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_67, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_69, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_70, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_71, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_72, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_77, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_78, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_80, (96, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_81, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_82, (144, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_83, (144, ), (1, ))
    assert_size_stride(primals_89, (144, ), (1, ))
    assert_size_stride(primals_95, (144, ), (1, ))
    assert_size_stride(primals_101, (144, ), (1, ))
    assert_size_stride(primals_107, (144, ), (1, ))
    assert_size_stride(primals_109, (96, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_110, (96, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_111, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_112, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_114, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_115, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_122, (192, ), (1, ))
    assert_size_stride(primals_128, (192, ), (1, ))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_140, (192, ), (1, ))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_158, (192, ), (1, ))
    assert_size_stride(primals_164, (192, ), (1, ))
    assert_size_stride(primals_166, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_167, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_168, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (160, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_171, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_172, (240, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_173, (240, ), (1, ))
    assert_size_stride(primals_179, (240, ), (1, ))
    assert_size_stride(primals_185, (240, ), (1, ))
    assert_size_stride(primals_191, (240, ), (1, ))
    assert_size_stride(primals_197, (240, ), (1, ))
    assert_size_stride(primals_203, (240, ), (1, ))
    assert_size_stride(primals_209, (240, ), (1, ))
    assert_size_stride(primals_211, (160, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_212, (160, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_213, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_312, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(convolution, (8, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(mul_7, (8, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(convolution_1, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(mul_15, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(convolution_2, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(mul_23, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(convolution_3, (8, 32, 128, 128), (524288, 16384, 128, 1))
    assert_size_stride(squeeze_10, (32, ), (1, ))
    assert_size_stride(add_19, (8, 32, 128, 128), (524288, 16384, 128, 1))
    assert_size_stride(convolution_4, (8, 128, 128, 128), (2097152, 16384, 128, 1))
    assert_size_stride(squeeze_13, (128, ), (1, ))
    assert_size_stride(mul_38, (8, 128, 128, 128), (2097152, 16384, 128, 1))
    assert_size_stride(convolution_5, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(squeeze_16, (128, ), (1, ))
    assert_size_stride(mul_46, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(convolution_6, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(add_34, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_7, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(mul_61, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_8, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_25, (256, ), (1, ))
    assert_size_stride(mul_69, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(add_50, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_10, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(mul_84, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_11, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(mul_92, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_12, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(squeeze_37, (64, ), (1, ))
    assert_size_stride(add_66, (8, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(convolution_13, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(squeeze_40, (256, ), (1, ))
    assert_size_stride(mul_107, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(convolution_14, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(squeeze_43, (256, ), (1, ))
    assert_size_stride(mul_115, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(convolution_15, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(squeeze_46, (96, ), (1, ))
    assert_size_stride(add_81, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(convolution_16, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(squeeze_49, (96, ), (1, ))
    assert_size_stride(mul_130, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(mul_131, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_3, (8192, 144), (144, 1))
    assert_size_stride(getitem_36, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_37, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_38, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_40, (32, 4, 256), (1024, 256, 1))
    assert_size_stride(getitem_41, (), ())
    assert_size_stride(getitem_42, (), ())
    assert_size_stride(view_7, (8192, 144), (144, 1))
    assert_size_stride(mul_133, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_9, (8192, 144), (144, 1))
    assert_size_stride(addmm_2, (8192, 288), (288, 1))
    assert_size_stride(view_11, (8192, 288), (288, 1))
    assert_size_stride(mul_136, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_13, (8192, 144), (144, 1))
    assert_size_stride(getitem_47, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_48, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_49, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_51, (32, 4, 256), (1024, 256, 1))
    assert_size_stride(getitem_52, (), ())
    assert_size_stride(getitem_53, (), ())
    assert_size_stride(view_17, (8192, 144), (144, 1))
    assert_size_stride(mul_138, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_19, (8192, 144), (144, 1))
    assert_size_stride(addmm_6, (8192, 288), (288, 1))
    assert_size_stride(view_21, (8192, 288), (288, 1))
    assert_size_stride(mul_141, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_25, (8, 144, 32, 32), (147456, 1024, 32, 1))
    assert_size_stride(convolution_18, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(squeeze_52, (96, ), (1, ))
    assert_size_stride(cat, (8, 192, 32, 32), (196608, 1024, 32, 1))
    assert_size_stride(convolution_19, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(squeeze_55, (96, ), (1, ))
    assert_size_stride(mul_158, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(convolution_20, (8, 384, 32, 32), (393216, 1024, 32, 1))
    assert_size_stride(squeeze_58, (384, ), (1, ))
    assert_size_stride(mul_166, (8, 384, 32, 32), (393216, 1024, 32, 1))
    assert_size_stride(convolution_21, (8, 384, 16, 16), (98304, 256, 16, 1))
    assert_size_stride(squeeze_61, (384, ), (1, ))
    assert_size_stride(mul_174, (8, 384, 16, 16), (98304, 256, 16, 1))
    assert_size_stride(convolution_22, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(add_125, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(convolution_23, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(mul_189, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(mul_190, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_29, (2048, 192), (192, 1))
    assert_size_stride(getitem_72, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_73, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_74, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_76, (32, 4, 64), (256, 64, 1))
    assert_size_stride(getitem_77, (), ())
    assert_size_stride(getitem_78, (), ())
    assert_size_stride(view_33, (2048, 192), (192, 1))
    assert_size_stride(mul_192, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_35, (2048, 192), (192, 1))
    assert_size_stride(addmm_10, (2048, 384), (384, 1))
    assert_size_stride(view_37, (2048, 384), (384, 1))
    assert_size_stride(mul_195, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_39, (2048, 192), (192, 1))
    assert_size_stride(getitem_83, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_84, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_85, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_87, (32, 4, 64), (256, 64, 1))
    assert_size_stride(getitem_88, (), ())
    assert_size_stride(getitem_89, (), ())
    assert_size_stride(view_43, (2048, 192), (192, 1))
    assert_size_stride(mul_197, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_45, (2048, 192), (192, 1))
    assert_size_stride(addmm_14, (2048, 384), (384, 1))
    assert_size_stride(view_47, (2048, 384), (384, 1))
    assert_size_stride(mul_200, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_49, (2048, 192), (192, 1))
    assert_size_stride(getitem_94, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_95, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_96, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_98, (32, 4, 64), (256, 64, 1))
    assert_size_stride(getitem_99, (), ())
    assert_size_stride(getitem_100, (), ())
    assert_size_stride(view_53, (2048, 192), (192, 1))
    assert_size_stride(mul_202, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_55, (2048, 192), (192, 1))
    assert_size_stride(addmm_18, (2048, 384), (384, 1))
    assert_size_stride(view_57, (2048, 384), (384, 1))
    assert_size_stride(mul_205, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_59, (2048, 192), (192, 1))
    assert_size_stride(getitem_105, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_106, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_107, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_109, (32, 4, 64), (256, 64, 1))
    assert_size_stride(getitem_110, (), ())
    assert_size_stride(getitem_111, (), ())
    assert_size_stride(view_63, (2048, 192), (192, 1))
    assert_size_stride(mul_207, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_65, (2048, 192), (192, 1))
    assert_size_stride(addmm_22, (2048, 384), (384, 1))
    assert_size_stride(view_67, (2048, 384), (384, 1))
    assert_size_stride(mul_210, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_71, (8, 192, 16, 16), (49152, 256, 16, 1))
    assert_size_stride(convolution_25, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(squeeze_70, (128, ), (1, ))
    assert_size_stride(cat_1, (8, 256, 16, 16), (65536, 256, 16, 1))
    assert_size_stride(convolution_26, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(squeeze_73, (128, ), (1, ))
    assert_size_stride(mul_227, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(convolution_27, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(squeeze_76, (512, ), (1, ))
    assert_size_stride(mul_235, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(convolution_28, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(squeeze_79, (512, ), (1, ))
    assert_size_stride(mul_243, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(convolution_29, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(squeeze_82, (160, ), (1, ))
    assert_size_stride(add_181, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(convolution_30, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(squeeze_85, (160, ), (1, ))
    assert_size_stride(mul_258, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(mul_259, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_75, (512, 240), (240, 1))
    assert_size_stride(getitem_130, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_131, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_132, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_134, (32, 4, 32), (128, 32, 1))
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(view_79, (512, 240), (240, 1))
    assert_size_stride(mul_261, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_81, (512, 240), (240, 1))
    assert_size_stride(addmm_26, (512, 480), (480, 1))
    assert_size_stride(view_83, (512, 480), (480, 1))
    assert_size_stride(mul_264, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_85, (512, 240), (240, 1))
    assert_size_stride(getitem_141, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_142, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_143, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_145, (32, 4, 32), (128, 32, 1))
    assert_size_stride(getitem_146, (), ())
    assert_size_stride(getitem_147, (), ())
    assert_size_stride(view_89, (512, 240), (240, 1))
    assert_size_stride(mul_266, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_91, (512, 240), (240, 1))
    assert_size_stride(addmm_30, (512, 480), (480, 1))
    assert_size_stride(view_93, (512, 480), (480, 1))
    assert_size_stride(mul_269, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_95, (512, 240), (240, 1))
    assert_size_stride(getitem_152, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_153, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_154, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_156, (32, 4, 32), (128, 32, 1))
    assert_size_stride(getitem_157, (), ())
    assert_size_stride(getitem_158, (), ())
    assert_size_stride(view_99, (512, 240), (240, 1))
    assert_size_stride(mul_271, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_101, (512, 240), (240, 1))
    assert_size_stride(addmm_34, (512, 480), (480, 1))
    assert_size_stride(view_103, (512, 480), (480, 1))
    assert_size_stride(mul_274, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_107, (8, 240, 8, 8), (15360, 64, 8, 1))
    assert_size_stride(convolution_32, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(squeeze_88, (160, ), (1, ))
    assert_size_stride(cat_2, (8, 320, 8, 8), (20480, 64, 8, 1))
    assert_size_stride(convolution_33, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(squeeze_91, (160, ), (1, ))
    assert_size_stride(mul_291, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(convolution_34, (8, 640, 8, 8), (40960, 64, 8, 1))
    assert_size_stride(squeeze_94, (640, ), (1, ))
    assert_size_stride(clone_64, (8, 640), (640, 1))
    assert_size_stride(permute_67, (1000, 640), (640, 1))
    assert_size_stride(mul_301, (8, 640, 8, 8), (40960, 64, 8, 1))
    assert_size_stride(unsqueeze_130, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(mul_313, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(unsqueeze_142, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(mul_325, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(unsqueeze_154, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(div_1, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_76, (240, 480), (480, 1))
    assert_size_stride(permute_81, (480, 240), (240, 1))
    assert_size_stride(div_2, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_85, (240, 240), (240, 1))
    assert_size_stride(alias_9, (32, 4, 16, 60), (3840, 60, 240, 1))
    assert_size_stride(permute_91, (720, 240), (240, 1))
    assert_size_stride(div_3, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_95, (240, 480), (480, 1))
    assert_size_stride(permute_100, (480, 240), (240, 1))
    assert_size_stride(div_4, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_104, (240, 240), (240, 1))
    assert_size_stride(alias_10, (32, 4, 16, 60), (3840, 60, 240, 1))
    assert_size_stride(permute_110, (720, 240), (240, 1))
    assert_size_stride(div_5, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_114, (240, 480), (480, 1))
    assert_size_stride(permute_119, (480, 240), (240, 1))
    assert_size_stride(div_6, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_123, (240, 240), (240, 1))
    assert_size_stride(alias_11, (32, 4, 16, 60), (3840, 60, 240, 1))
    assert_size_stride(permute_129, (720, 240), (240, 1))
    assert_size_stride(div_7, (32, 16, 1), (16, 1, 1))
    assert_size_stride(mul_395, (8, 160, 8, 8), (10240, 64, 8, 1))
    assert_size_stride(unsqueeze_166, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(mul_416, (8, 512, 8, 8), (32768, 64, 8, 1))
    assert_size_stride(unsqueeze_190, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_428, (8, 512, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(unsqueeze_202, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_440, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(unsqueeze_214, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_452, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(unsqueeze_226, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_8, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_142, (192, 384), (384, 1))
    assert_size_stride(permute_147, (384, 192), (192, 1))
    assert_size_stride(div_9, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_151, (192, 192), (192, 1))
    assert_size_stride(alias_12, (32, 4, 64, 48), (12288, 48, 192, 1))
    assert_size_stride(permute_157, (576, 192), (192, 1))
    assert_size_stride(div_10, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_161, (192, 384), (384, 1))
    assert_size_stride(permute_166, (384, 192), (192, 1))
    assert_size_stride(div_11, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_170, (192, 192), (192, 1))
    assert_size_stride(alias_13, (32, 4, 64, 48), (12288, 48, 192, 1))
    assert_size_stride(permute_176, (576, 192), (192, 1))
    assert_size_stride(div_12, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_180, (192, 384), (384, 1))
    assert_size_stride(permute_185, (384, 192), (192, 1))
    assert_size_stride(div_13, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_189, (192, 192), (192, 1))
    assert_size_stride(alias_14, (32, 4, 64, 48), (12288, 48, 192, 1))
    assert_size_stride(permute_195, (576, 192), (192, 1))
    assert_size_stride(div_14, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_199, (192, 384), (384, 1))
    assert_size_stride(permute_204, (384, 192), (192, 1))
    assert_size_stride(div_15, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_208, (192, 192), (192, 1))
    assert_size_stride(alias_15, (32, 4, 64, 48), (12288, 48, 192, 1))
    assert_size_stride(permute_214, (576, 192), (192, 1))
    assert_size_stride(div_16, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_539, (8, 128, 16, 16), (32768, 256, 16, 1))
    assert_size_stride(unsqueeze_238, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_560, (8, 384, 16, 16), (98304, 256, 16, 1))
    assert_size_stride(unsqueeze_262, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(mul_572, (8, 384, 32, 32), (393216, 1024, 32, 1))
    assert_size_stride(unsqueeze_274, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(mul_584, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(unsqueeze_286, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_596, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(unsqueeze_298, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(div_17, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_227, (144, 288), (288, 1))
    assert_size_stride(permute_232, (288, 144), (144, 1))
    assert_size_stride(div_18, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_236, (144, 144), (144, 1))
    assert_size_stride(alias_16, (32, 4, 256, 36), (36864, 36, 144, 1))
    assert_size_stride(permute_242, (432, 144), (144, 1))
    assert_size_stride(div_19, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_246, (144, 288), (288, 1))
    assert_size_stride(permute_251, (288, 144), (144, 1))
    assert_size_stride(div_20, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_255, (144, 144), (144, 1))
    assert_size_stride(alias_17, (32, 4, 256, 36), (36864, 36, 144, 1))
    assert_size_stride(permute_261, (432, 144), (144, 1))
    assert_size_stride(div_21, (32, 256, 1), (256, 1, 1))
    assert_size_stride(mul_649, (8, 96, 32, 32), (98304, 1024, 32, 1))
    assert_size_stride(unsqueeze_310, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_670, (8, 256, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(unsqueeze_334, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_682, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(unsqueeze_346, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_703, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(unsqueeze_370, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_715, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(unsqueeze_382, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_736, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(unsqueeze_406, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_748, (8, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(unsqueeze_418, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_769, (8, 128, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(unsqueeze_442, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_781, (8, 128, 128, 128), (2097152, 16384, 128, 1))
    assert_size_stride(unsqueeze_454, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_802, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(unsqueeze_478, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_814, (8, 64, 128, 128), (1048576, 16384, 128, 1))
    assert_size_stride(unsqueeze_490, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_826, (8, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(unsqueeze_502, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_67, out=buf0)
        del permute_67
        buf1 = empty((1000, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_64, out=buf1)
        del clone_64
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((640, ), device='cuda', dtype=torch.float32)
        buf4 = empty((640, ), device='cuda', dtype=torch.float32)
        buf5 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_1.run(buf0, mul_301, convolution_34, unsqueeze_130, squeeze_94, buf3, buf4, buf5, 640, 512, grid=grid(640), stream=stream0)
        buf6 = empty((8, 640, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_2.run(buf0, mul_301, convolution_34, unsqueeze_130, buf4, squeeze_94, buf3, primals_63, buf6, 327680, grid=grid(327680), stream=stream0)
        del buf0
        del buf4
        del convolution_34
        del mul_301
        del primals_63
        del squeeze_94
        del unsqueeze_130
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        buf7 = aten.convolution_backward(buf6, mul_291, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf6
        del mul_291
        del primals_213
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        buf10 = empty((160, ), device='cuda', dtype=torch.float32)
        buf11 = empty((160, ), device='cuda', dtype=torch.float32)
        buf12 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_3.run(buf8, mul_313, convolution_33, unsqueeze_142, squeeze_91, buf10, buf11, buf12, 160, 512, grid=grid(160), stream=stream0)
        buf13 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_4.run(buf13, mul_313, convolution_33, unsqueeze_142, buf11, squeeze_91, buf10, primals_61, 81920, grid=grid(81920), stream=stream0)
        del convolution_33
        del mul_313
        del primals_61
        del squeeze_91
        del unsqueeze_142
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf14 = aten.convolution_backward(buf13, cat_2, primals_212, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_2
        del primals_212
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = buf11; del buf11  # reuse
        buf18 = empty((160, ), device='cuda', dtype=torch.float32)
        buf19 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_5.run(buf15, mul_325, convolution_32, unsqueeze_154, squeeze_88, buf17, buf18, buf19, 160, 512, grid=grid(160), stream=stream0)
        buf20 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_6.run(buf15, mul_325, convolution_32, unsqueeze_154, buf18, squeeze_88, buf17, primals_59, buf20, 81920, grid=grid(81920), stream=stream0)
        del convolution_32
        del mul_325
        del primals_59
        del squeeze_88
        del unsqueeze_154
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf21 = aten.convolution_backward(buf20, view_107, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf20
        del primals_211
        del view_107
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = empty_strided((32, 16, 1, 2), (1, 32, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_7.run(buf22, primals_209, buf24, 1024, 120, grid=grid(1024), stream=stream0)
        buf25 = empty_strided((32, 16, 1), (16, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_8.run(buf24, buf25, 512, 2, grid=grid(512), stream=stream0)
        buf26 = reinterpret_tensor(buf24, (32, 16, 1, 2), (32, 2, 1024, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_9.run(buf22, primals_209, mul_274, buf26, 1024, 120, grid=grid(1024), stream=stream0)
        buf27 = empty_strided((32, 16, 1), (16, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_10.run(buf26, buf27, 512, 2, grid=grid(512), stream=stream0)
        buf28 = empty((32, 16, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_backward_11.run(div_1, buf22, primals_209, buf25, mul_274, buf27, buf28, 32, 3840, grid=grid(32, 3840), stream=stream0)
        del div_1
        del primals_209
        buf29 = empty_strided((240, 4), (1, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf22, mul_274, buf29, 960, 128, grid=grid(960), stream=stream0)
        del mul_274
        buf30 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf29, buf30, 240, 4, grid=grid(240), stream=stream0)
        buf31 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_14.run(buf22, buf31, 240, 512, grid=grid(240), stream=stream0)
        buf32 = empty((512, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 240), (240, 1), 0), permute_76, out=buf32)
        del permute_76
        buf33 = empty((240, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (240, 512), (1, 240), 0), view_103, out=buf33)
        del view_103
        buf34 = reinterpret_tensor(buf29, (1, 240, 4), (960, 1, 240), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf28, buf34, 960, 128, grid=grid(960), stream=stream0)
        buf35 = empty((1, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_13.run(buf34, buf35, 240, 4, grid=grid(240), stream=stream0)
        buf36 = reinterpret_tensor(buf32, (32, 16, 480), (7680, 480, 1), 0); del buf32  # reuse
        # Source Nodes: [x_325], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_16.run(buf36, addmm_34, 245760, grid=grid(245760), stream=stream0)
        del addmm_34
        buf37 = reinterpret_tensor(buf22, (512, 240), (240, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (512, 480), (480, 1), 0), permute_81, out=buf37)
        del permute_81
        buf38 = empty((480, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (480, 512), (1, 480), 0), view_101, out=buf38)
        del view_101
        buf39 = empty_strided((1, 480, 4), (1920, 1, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf36, buf39, 1920, 128, grid=grid(1920), stream=stream0)
        buf40 = empty((1, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf39, buf40, 480, 4, grid=grid(480), stream=stream0)
        buf47 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf47, buf37, primals_203, mul_271, div_2, 512, 240, grid=grid(512), stream=stream0)
        del div_2
        del primals_203
        buf43 = reinterpret_tensor(buf34, (240, 4), (1, 240), 0); del buf34  # reuse
        buf45 = empty_strided((240, 4), (1, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf37, mul_271, buf43, buf45, 960, 128, grid=grid(960), stream=stream0)
        del mul_271
        buf44 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf43, buf44, 240, 4, grid=grid(240), stream=stream0)
        buf46 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf45, buf46, 240, 4, grid=grid(240), stream=stream0)
        buf48 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (512, 240), (240, 1), 0), permute_85, out=buf48)
        del permute_85
        buf49 = empty((240, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (240, 512), (1, 240), 0), view_99, out=buf49)
        del view_99
        buf50 = reinterpret_tensor(buf45, (1, 240, 4), (960, 1, 240), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf47, buf50, 960, 128, grid=grid(960), stream=stream0)
        buf51 = empty((1, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_13.run(buf50, buf51, 240, 4, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf52 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf48, (32, 4, 16, 60), (3840, 60, 240, 1), 0), getitem_152, getitem_153, getitem_154, None, alias_9, getitem_156, getitem_157, getitem_158, 0.0, [True, True, True, False])
        del alias_9
        del buf48
        del getitem_152
        del getitem_153
        del getitem_154
        del getitem_156
        del getitem_157
        del getitem_158
        buf53 = buf52[0]
        buf54 = buf52[1]
        buf55 = buf52[2]
        del buf52
        buf56 = empty((32, 16, 3, 4, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf53, buf54, buf55, buf56, 368640, grid=grid(368640), stream=stream0)
        del buf53
        del buf54
        buf57 = reinterpret_tensor(buf55, (512, 240), (240, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (512, 720), (720, 1), 0), permute_91, out=buf57)
        del permute_91
        buf58 = empty((720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (720, 512), (1, 720), 0), view_95, out=buf58)
        del view_95
        buf59 = empty_strided((1, 720, 4), (2880, 1, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf56, buf59, 2880, 128, grid=grid(2880), stream=stream0)
        buf60 = empty((1, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf59, buf60, 720, 4, grid=grid(720), stream=stream0)
        buf67 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf67, buf57, primals_197, mul_269, div_3, 512, 240, grid=grid(512), stream=stream0)
        del div_3
        del primals_197
        buf63 = reinterpret_tensor(buf50, (240, 4), (1, 240), 0); del buf50  # reuse
        buf65 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf57, mul_269, buf63, buf65, 960, 128, grid=grid(960), stream=stream0)
        del mul_269
        buf64 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf63, buf64, 240, 4, grid=grid(240), stream=stream0)
        buf66 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf65, buf66, 240, 4, grid=grid(240), stream=stream0)
        buf68 = reinterpret_tensor(buf36, (512, 480), (480, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (512, 240), (240, 1), 0), permute_95, out=buf68)
        del permute_95
        buf69 = empty((240, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (240, 512), (1, 240), 0), view_93, out=buf69)
        del view_93
        buf70 = reinterpret_tensor(buf65, (1, 240, 4), (960, 1, 240), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf67, buf70, 960, 128, grid=grid(960), stream=stream0)
        buf71 = empty((1, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_13.run(buf70, buf71, 240, 4, grid=grid(240), stream=stream0)
        buf72 = reinterpret_tensor(buf68, (32, 16, 480), (7680, 480, 1), 0); del buf68  # reuse
        # Source Nodes: [x_313], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_16.run(buf72, addmm_30, 245760, grid=grid(245760), stream=stream0)
        del addmm_30
        buf73 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (512, 480), (480, 1), 0), permute_100, out=buf73)
        del permute_100
        buf74 = empty((480, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (480, 512), (1, 480), 0), view_91, out=buf74)
        del view_91
        buf75 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf72, buf75, 1920, 128, grid=grid(1920), stream=stream0)
        buf76 = empty((1, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf75, buf76, 480, 4, grid=grid(480), stream=stream0)
        buf83 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf83, buf73, primals_191, mul_266, div_4, 512, 240, grid=grid(512), stream=stream0)
        del div_4
        del primals_191
        buf79 = reinterpret_tensor(buf70, (240, 4), (1, 240), 0); del buf70  # reuse
        buf81 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf73, mul_266, buf79, buf81, 960, 128, grid=grid(960), stream=stream0)
        del mul_266
        buf80 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf79, buf80, 240, 4, grid=grid(240), stream=stream0)
        buf82 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf81, buf82, 240, 4, grid=grid(240), stream=stream0)
        buf84 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (512, 240), (240, 1), 0), permute_104, out=buf84)
        del permute_104
        buf85 = empty((240, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (240, 512), (1, 240), 0), view_89, out=buf85)
        del view_89
        buf86 = reinterpret_tensor(buf81, (1, 240, 4), (960, 1, 240), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf83, buf86, 960, 128, grid=grid(960), stream=stream0)
        buf87 = empty((1, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_13.run(buf86, buf87, 240, 4, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf88 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf84, (32, 4, 16, 60), (3840, 60, 240, 1), 0), getitem_141, getitem_142, getitem_143, None, alias_10, getitem_145, getitem_146, getitem_147, 0.0, [True, True, True, False])
        del alias_10
        del buf84
        del getitem_141
        del getitem_142
        del getitem_143
        del getitem_145
        del getitem_146
        del getitem_147
        buf89 = buf88[0]
        buf90 = buf88[1]
        buf91 = buf88[2]
        del buf88
        buf92 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf89, buf90, buf91, buf92, 368640, grid=grid(368640), stream=stream0)
        del buf89
        del buf90
        buf93 = reinterpret_tensor(buf91, (512, 240), (240, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (512, 720), (720, 1), 0), permute_110, out=buf93)
        del permute_110
        buf94 = empty((720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (720, 512), (1, 720), 0), view_85, out=buf94)
        del view_85
        buf95 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf92, buf95, 2880, 128, grid=grid(2880), stream=stream0)
        buf96 = empty((1, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf95, buf96, 720, 4, grid=grid(720), stream=stream0)
        buf103 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf103, buf93, primals_185, mul_264, div_5, 512, 240, grid=grid(512), stream=stream0)
        del div_5
        del primals_185
        buf99 = reinterpret_tensor(buf86, (240, 4), (1, 240), 0); del buf86  # reuse
        buf101 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf93, mul_264, buf99, buf101, 960, 128, grid=grid(960), stream=stream0)
        del mul_264
        buf100 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf99, buf100, 240, 4, grid=grid(240), stream=stream0)
        buf102 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf101, buf102, 240, 4, grid=grid(240), stream=stream0)
        buf104 = reinterpret_tensor(buf72, (512, 480), (480, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (512, 240), (240, 1), 0), permute_114, out=buf104)
        del permute_114
        buf105 = empty((240, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (240, 512), (1, 240), 0), view_83, out=buf105)
        del view_83
        buf106 = reinterpret_tensor(buf101, (1, 240, 4), (960, 1, 240), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf103, buf106, 960, 128, grid=grid(960), stream=stream0)
        buf107 = empty((1, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_13.run(buf106, buf107, 240, 4, grid=grid(240), stream=stream0)
        buf108 = reinterpret_tensor(buf104, (32, 16, 480), (7680, 480, 1), 0); del buf104  # reuse
        # Source Nodes: [x_301], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_16.run(buf108, addmm_26, 245760, grid=grid(245760), stream=stream0)
        del addmm_26
        buf109 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (512, 480), (480, 1), 0), permute_119, out=buf109)
        del permute_119
        buf110 = empty((480, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (480, 512), (1, 480), 0), view_81, out=buf110)
        del view_81
        buf111 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf108, buf111, 1920, 128, grid=grid(1920), stream=stream0)
        del buf108
        buf112 = empty((1, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf111, buf112, 480, 4, grid=grid(480), stream=stream0)
        del buf111
        buf119 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_19.run(buf119, buf109, primals_179, mul_261, div_6, 512, 240, grid=grid(512), stream=stream0)
        del div_6
        del primals_179
        buf115 = reinterpret_tensor(buf106, (240, 4), (1, 240), 0); del buf106  # reuse
        buf117 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf109, mul_261, buf115, buf117, 960, 128, grid=grid(960), stream=stream0)
        del mul_261
        buf116 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf115, buf116, 240, 4, grid=grid(240), stream=stream0)
        buf118 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf117, buf118, 240, 4, grid=grid(240), stream=stream0)
        buf120 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (512, 240), (240, 1), 0), permute_123, out=buf120)
        del permute_123
        buf121 = empty((240, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (240, 512), (1, 240), 0), view_79, out=buf121)
        del view_79
        buf122 = reinterpret_tensor(buf117, (1, 240, 4), (960, 1, 240), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf119, buf122, 960, 128, grid=grid(960), stream=stream0)
        buf123 = empty((1, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_13.run(buf122, buf123, 240, 4, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf124 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf120, (32, 4, 16, 60), (3840, 60, 240, 1), 0), getitem_130, getitem_131, getitem_132, None, alias_11, getitem_134, getitem_135, getitem_136, 0.0, [True, True, True, False])
        del alias_11
        del buf120
        del getitem_130
        del getitem_131
        del getitem_132
        del getitem_134
        del getitem_135
        del getitem_136
        buf125 = buf124[0]
        buf126 = buf124[1]
        buf127 = buf124[2]
        del buf124
        buf128 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf125, buf126, buf127, buf128, 368640, grid=grid(368640), stream=stream0)
        del buf125
        buf129 = reinterpret_tensor(buf127, (512, 240), (240, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 720), (720, 1), 0), permute_129, out=buf129)
        del permute_129
        buf130 = empty((720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (720, 512), (1, 720), 0), view_75, out=buf130)
        del view_75
        buf131 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf128, buf131, 2880, 128, grid=grid(2880), stream=stream0)
        del buf128
        buf132 = empty((1, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_23.run(buf131, buf132, 720, 4, grid=grid(720), stream=stream0)
        del buf131
        buf133 = buf27; del buf27  # reuse
        buf134 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_24.run(buf129, primals_173, mul_259, buf133, buf134, 512, 240, grid=grid(512), stream=stream0)
        buf135 = reinterpret_tensor(buf122, (240, 4), (1, 240), 0); del buf122  # reuse
        buf137 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_20.run(buf129, mul_259, buf135, buf137, 960, 128, grid=grid(960), stream=stream0)
        buf136 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf135, buf136, 240, 4, grid=grid(240), stream=stream0)
        del buf135
        buf138 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf137, buf138, 240, 4, grid=grid(240), stream=stream0)
        del buf137
        buf139 = reinterpret_tensor(buf126, (8, 240, 8, 8), (15360, 64, 8, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_25.run(buf119, div_7, buf129, primals_173, buf133, mul_259, buf134, buf139, 512, 240, grid=grid(512, 240), stream=stream0)
        del buf119
        del buf129
        del div_7
        del mul_259
        del primals_173
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf140 = aten.convolution_backward(buf139, mul_258, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf139
        del mul_258
        del primals_172
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        buf143 = buf18; del buf18  # reuse
        buf144 = empty((160, ), device='cuda', dtype=torch.float32)
        buf145 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_3.run(buf141, mul_395, convolution_30, unsqueeze_166, squeeze_85, buf143, buf144, buf145, 160, 512, grid=grid(160), stream=stream0)
        buf146 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_4.run(buf146, mul_395, convolution_30, unsqueeze_166, buf144, squeeze_85, buf143, primals_57, 81920, grid=grid(81920), stream=stream0)
        del convolution_30
        del mul_395
        del primals_57
        del squeeze_85
        del unsqueeze_166
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf147 = aten.convolution_backward(buf146, add_181, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_181
        del buf146
        del primals_171
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf150 = buf144; del buf144  # reuse
        buf151 = empty((160, ), device='cuda', dtype=torch.float32)
        buf152 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_26.run(buf15, buf148, convolution_29, unsqueeze_178, squeeze_82, buf150, buf151, buf152, 160, 512, grid=grid(160), stream=stream0)
        buf153 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_27.run(buf153, buf15, convolution_29, unsqueeze_178, buf151, squeeze_82, buf150, primals_55, 81920, grid=grid(81920), stream=stream0)
        del buf15
        del buf151
        del convolution_29
        del primals_55
        del squeeze_82
        del unsqueeze_178
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf154 = aten.convolution_backward(buf153, mul_243, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf153
        del mul_243
        del primals_170
        buf155 = buf154[0]
        buf156 = buf154[1]
        del buf154
        buf157 = reinterpret_tensor(buf134, (512, ), (1, ), 0); del buf134  # reuse
        buf158 = reinterpret_tensor(buf133, (512, ), (1, ), 0); del buf133  # reuse
        buf159 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_28.run(buf155, mul_416, convolution_28, unsqueeze_190, squeeze_79, buf157, buf158, buf159, 512, 512, grid=grid(512), stream=stream0)
        buf160 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_29.run(buf160, mul_416, convolution_28, unsqueeze_190, buf158, squeeze_79, buf157, primals_53, 262144, grid=grid(262144), stream=stream0)
        del convolution_28
        del mul_416
        del primals_53
        del squeeze_79
        del unsqueeze_190
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf161 = aten.convolution_backward(buf160, mul_235, primals_169, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False])
        del buf160
        del mul_235
        del primals_169
        buf162 = buf161[0]
        buf163 = buf161[1]
        del buf161
        buf164 = buf158; del buf158  # reuse
        buf165 = empty((512, ), device='cuda', dtype=torch.float32)
        buf166 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_30.run(buf162, mul_428, convolution_27, unsqueeze_202, squeeze_76, buf164, buf165, buf166, 512, 2048, grid=grid(512), stream=stream0)
        buf167 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_31.run(buf167, mul_428, convolution_27, unsqueeze_202, buf165, squeeze_76, buf164, primals_51, 1048576, grid=grid(1048576), stream=stream0)
        del convolution_27
        del mul_428
        del primals_51
        del squeeze_76
        del unsqueeze_202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf168 = aten.convolution_backward(buf167, mul_227, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf167
        del mul_227
        del primals_168
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = empty((128, ), device='cuda', dtype=torch.float32)
        buf172 = empty((128, ), device='cuda', dtype=torch.float32)
        buf173 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_32.run(buf169, mul_440, convolution_26, unsqueeze_214, squeeze_73, buf171, buf172, buf173, 128, 2048, grid=grid(128), stream=stream0)
        buf174 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_33.run(buf174, mul_440, convolution_26, unsqueeze_214, buf172, squeeze_73, buf171, primals_49, 262144, grid=grid(262144), stream=stream0)
        del convolution_26
        del mul_440
        del primals_49
        del squeeze_73
        del unsqueeze_214
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf175 = aten.convolution_backward(buf174, cat_1, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_167
        buf176 = buf175[0]
        buf177 = buf175[1]
        del buf175
        buf178 = buf172; del buf172  # reuse
        buf179 = empty((128, ), device='cuda', dtype=torch.float32)
        buf180 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_34.run(buf176, mul_452, convolution_25, unsqueeze_226, squeeze_70, buf178, buf179, buf180, 128, 2048, grid=grid(128), stream=stream0)
        buf181 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_35.run(buf176, mul_452, convolution_25, unsqueeze_226, buf179, squeeze_70, buf178, primals_47, buf181, 262144, grid=grid(262144), stream=stream0)
        del convolution_25
        del mul_452
        del primals_47
        del squeeze_70
        del unsqueeze_226
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf182 = aten.convolution_backward(buf181, view_71, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf181
        del primals_166
        del view_71
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        buf185 = empty_strided((32, 64, 1, 2), (1, 32, 4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf183, primals_164, buf185, 4096, 96, grid=grid(4096), stream=stream0)
        buf186 = empty_strided((32, 64, 1), (64, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf185, buf186, 2048, 2, grid=grid(2048), stream=stream0)
        buf187 = reinterpret_tensor(buf185, (32, 64, 1, 2), (128, 2, 4096, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf183, primals_164, mul_210, buf187, 4096, 96, grid=grid(4096), stream=stream0)
        buf188 = empty_strided((32, 64, 1), (64, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf187, buf188, 2048, 2, grid=grid(2048), stream=stream0)
        del buf187
        buf189 = empty((32, 64, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_backward_40.run(div_8, buf183, primals_164, buf186, mul_210, buf188, buf189, 32, 12288, grid=grid(32, 12288), stream=stream0)
        del div_8
        del primals_164
        buf190 = empty((192, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf183, mul_210, buf190, 3072, 128, grid=grid(3072), stream=stream0)
        del mul_210
        buf191 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_42.run(buf190, buf191, 192, 16, grid=grid(192), stream=stream0)
        buf192 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_43.run(buf183, buf192, 192, 2048, grid=grid(192), stream=stream0)
        buf193 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (2048, 192), (192, 1), 0), permute_142, out=buf193)
        del permute_142
        buf194 = empty((192, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (192, 2048), (1, 192), 0), view_67, out=buf194)
        del view_67
        buf195 = reinterpret_tensor(buf190, (1, 192, 16), (3072, 1, 192), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf189, buf195, 3072, 128, grid=grid(3072), stream=stream0)
        buf196 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf195, buf196, 192, 16, grid=grid(192), stream=stream0)
        buf197 = reinterpret_tensor(buf193, (32, 64, 384), (24576, 384, 1), 0); del buf193  # reuse
        # Source Nodes: [x_241], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_46.run(buf197, addmm_22, 786432, grid=grid(786432), stream=stream0)
        del addmm_22
        buf198 = reinterpret_tensor(buf183, (2048, 192), (192, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (2048, 384), (384, 1), 0), permute_147, out=buf198)
        del permute_147
        buf199 = empty((384, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (384, 2048), (1, 384), 0), view_65, out=buf199)
        del view_65
        buf200 = empty_strided((1, 384, 16), (6144, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_47.run(buf197, buf200, 6144, 128, grid=grid(6144), stream=stream0)
        buf201 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_48.run(buf200, buf201, 384, 16, grid=grid(384), stream=stream0)
        buf208 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_49.run(buf208, buf198, primals_158, mul_207, div_9, 2048, 192, grid=grid(2048), stream=stream0)
        del div_9
        del primals_158
        buf204 = reinterpret_tensor(buf195, (192, 16), (1, 192), 0); del buf195  # reuse
        buf206 = empty_strided((192, 16), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf198, mul_207, buf204, buf206, 3072, 128, grid=grid(3072), stream=stream0)
        del mul_207
        buf205 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf204, buf205, 192, 16, grid=grid(192), stream=stream0)
        buf207 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf206, buf207, 192, 16, grid=grid(192), stream=stream0)
        buf209 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (2048, 192), (192, 1), 0), permute_151, out=buf209)
        del permute_151
        buf210 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (192, 2048), (1, 192), 0), view_63, out=buf210)
        del view_63
        buf211 = reinterpret_tensor(buf206, (1, 192, 16), (3072, 1, 192), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf208, buf211, 3072, 128, grid=grid(3072), stream=stream0)
        buf212 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf211, buf212, 192, 16, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf213 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf209, (32, 4, 64, 48), (12288, 48, 192, 1), 0), getitem_105, getitem_106, getitem_107, None, alias_12, getitem_109, getitem_110, getitem_111, 0.0, [True, True, True, False])
        del alias_12
        del buf209
        del getitem_105
        del getitem_106
        del getitem_107
        del getitem_109
        del getitem_110
        del getitem_111
        buf214 = buf213[0]
        buf215 = buf213[1]
        buf216 = buf213[2]
        del buf213
        buf217 = empty((32, 64, 3, 4, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf214, buf215, buf216, buf217, 1179648, grid=grid(1179648), stream=stream0)
        del buf214
        del buf215
        buf218 = reinterpret_tensor(buf216, (2048, 192), (192, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (2048, 576), (576, 1), 0), permute_157, out=buf218)
        del permute_157
        buf219 = empty((576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (576, 2048), (1, 576), 0), view_59, out=buf219)
        del view_59
        buf220 = empty_strided((1, 576, 16), (9216, 1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_52.run(buf217, buf220, 9216, 128, grid=grid(9216), stream=stream0)
        buf221 = empty((1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_53.run(buf220, buf221, 576, 16, grid=grid(576), stream=stream0)
        buf228 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_49.run(buf228, buf218, primals_152, mul_205, div_10, 2048, 192, grid=grid(2048), stream=stream0)
        del div_10
        del primals_152
        buf224 = reinterpret_tensor(buf211, (192, 16), (1, 192), 0); del buf211  # reuse
        buf226 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf218, mul_205, buf224, buf226, 3072, 128, grid=grid(3072), stream=stream0)
        del mul_205
        buf225 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf224, buf225, 192, 16, grid=grid(192), stream=stream0)
        buf227 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf226, buf227, 192, 16, grid=grid(192), stream=stream0)
        buf229 = reinterpret_tensor(buf197, (2048, 384), (384, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (2048, 192), (192, 1), 0), permute_161, out=buf229)
        del permute_161
        buf230 = empty((192, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (192, 2048), (1, 192), 0), view_57, out=buf230)
        del view_57
        buf231 = reinterpret_tensor(buf226, (1, 192, 16), (3072, 1, 192), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf228, buf231, 3072, 128, grid=grid(3072), stream=stream0)
        buf232 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf231, buf232, 192, 16, grid=grid(192), stream=stream0)
        buf233 = reinterpret_tensor(buf229, (32, 64, 384), (24576, 384, 1), 0); del buf229  # reuse
        # Source Nodes: [x_229], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_46.run(buf233, addmm_18, 786432, grid=grid(786432), stream=stream0)
        del addmm_18
        buf234 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (2048, 384), (384, 1), 0), permute_166, out=buf234)
        del permute_166
        buf235 = empty((384, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (384, 2048), (1, 384), 0), view_55, out=buf235)
        del view_55
        buf236 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_47.run(buf233, buf236, 6144, 128, grid=grid(6144), stream=stream0)
        buf237 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_48.run(buf236, buf237, 384, 16, grid=grid(384), stream=stream0)
        buf244 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_49.run(buf244, buf234, primals_146, mul_202, div_11, 2048, 192, grid=grid(2048), stream=stream0)
        del div_11
        del primals_146
        buf240 = reinterpret_tensor(buf231, (192, 16), (1, 192), 0); del buf231  # reuse
        buf242 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf234, mul_202, buf240, buf242, 3072, 128, grid=grid(3072), stream=stream0)
        del mul_202
        buf241 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf240, buf241, 192, 16, grid=grid(192), stream=stream0)
        buf243 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf242, buf243, 192, 16, grid=grid(192), stream=stream0)
        buf245 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (2048, 192), (192, 1), 0), permute_170, out=buf245)
        del permute_170
        buf246 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (192, 2048), (1, 192), 0), view_53, out=buf246)
        del view_53
        buf247 = reinterpret_tensor(buf242, (1, 192, 16), (3072, 1, 192), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf244, buf247, 3072, 128, grid=grid(3072), stream=stream0)
        buf248 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf247, buf248, 192, 16, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf249 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf245, (32, 4, 64, 48), (12288, 48, 192, 1), 0), getitem_94, getitem_95, getitem_96, None, alias_13, getitem_98, getitem_99, getitem_100, 0.0, [True, True, True, False])
        del alias_13
        del buf245
        del getitem_100
        del getitem_94
        del getitem_95
        del getitem_96
        del getitem_98
        del getitem_99
        buf250 = buf249[0]
        buf251 = buf249[1]
        buf252 = buf249[2]
        del buf249
        buf253 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf250, buf251, buf252, buf253, 1179648, grid=grid(1179648), stream=stream0)
        del buf250
        del buf251
        buf254 = reinterpret_tensor(buf252, (2048, 192), (192, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (2048, 576), (576, 1), 0), permute_176, out=buf254)
        del permute_176
        buf255 = empty((576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (576, 2048), (1, 576), 0), view_49, out=buf255)
        del view_49
        buf256 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_52.run(buf253, buf256, 9216, 128, grid=grid(9216), stream=stream0)
        buf257 = empty((1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_53.run(buf256, buf257, 576, 16, grid=grid(576), stream=stream0)
        buf264 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_49.run(buf264, buf254, primals_140, mul_200, div_12, 2048, 192, grid=grid(2048), stream=stream0)
        del div_12
        del primals_140
        buf260 = reinterpret_tensor(buf247, (192, 16), (1, 192), 0); del buf247  # reuse
        buf262 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf254, mul_200, buf260, buf262, 3072, 128, grid=grid(3072), stream=stream0)
        del mul_200
        buf261 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf260, buf261, 192, 16, grid=grid(192), stream=stream0)
        buf263 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf262, buf263, 192, 16, grid=grid(192), stream=stream0)
        buf265 = reinterpret_tensor(buf233, (2048, 384), (384, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (2048, 192), (192, 1), 0), permute_180, out=buf265)
        del permute_180
        buf266 = empty((192, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (192, 2048), (1, 192), 0), view_47, out=buf266)
        del view_47
        buf267 = reinterpret_tensor(buf262, (1, 192, 16), (3072, 1, 192), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf264, buf267, 3072, 128, grid=grid(3072), stream=stream0)
        buf268 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf267, buf268, 192, 16, grid=grid(192), stream=stream0)
        buf269 = reinterpret_tensor(buf265, (32, 64, 384), (24576, 384, 1), 0); del buf265  # reuse
        # Source Nodes: [x_217], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_46.run(buf269, addmm_14, 786432, grid=grid(786432), stream=stream0)
        del addmm_14
        buf270 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (2048, 384), (384, 1), 0), permute_185, out=buf270)
        del permute_185
        buf271 = empty((384, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (384, 2048), (1, 384), 0), view_45, out=buf271)
        del view_45
        buf272 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_47.run(buf269, buf272, 6144, 128, grid=grid(6144), stream=stream0)
        buf273 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_48.run(buf272, buf273, 384, 16, grid=grid(384), stream=stream0)
        buf280 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_49.run(buf280, buf270, primals_134, mul_197, div_13, 2048, 192, grid=grid(2048), stream=stream0)
        del div_13
        del primals_134
        buf276 = reinterpret_tensor(buf267, (192, 16), (1, 192), 0); del buf267  # reuse
        buf278 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf270, mul_197, buf276, buf278, 3072, 128, grid=grid(3072), stream=stream0)
        del mul_197
        buf277 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf276, buf277, 192, 16, grid=grid(192), stream=stream0)
        buf279 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf278, buf279, 192, 16, grid=grid(192), stream=stream0)
        buf281 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (2048, 192), (192, 1), 0), permute_189, out=buf281)
        del permute_189
        buf282 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (192, 2048), (1, 192), 0), view_43, out=buf282)
        del view_43
        buf283 = reinterpret_tensor(buf278, (1, 192, 16), (3072, 1, 192), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf280, buf283, 3072, 128, grid=grid(3072), stream=stream0)
        buf284 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf283, buf284, 192, 16, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf285 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf281, (32, 4, 64, 48), (12288, 48, 192, 1), 0), getitem_83, getitem_84, getitem_85, None, alias_14, getitem_87, getitem_88, getitem_89, 0.0, [True, True, True, False])
        del alias_14
        del buf281
        del getitem_83
        del getitem_84
        del getitem_85
        del getitem_87
        del getitem_88
        del getitem_89
        buf286 = buf285[0]
        buf287 = buf285[1]
        buf288 = buf285[2]
        del buf285
        buf289 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf286, buf287, buf288, buf289, 1179648, grid=grid(1179648), stream=stream0)
        del buf286
        del buf287
        buf290 = reinterpret_tensor(buf288, (2048, 192), (192, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (2048, 576), (576, 1), 0), permute_195, out=buf290)
        del permute_195
        buf291 = empty((576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (576, 2048), (1, 576), 0), view_39, out=buf291)
        del view_39
        buf292 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_52.run(buf289, buf292, 9216, 128, grid=grid(9216), stream=stream0)
        buf293 = empty((1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_53.run(buf292, buf293, 576, 16, grid=grid(576), stream=stream0)
        buf300 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_49.run(buf300, buf290, primals_128, mul_195, div_14, 2048, 192, grid=grid(2048), stream=stream0)
        del div_14
        del primals_128
        buf296 = reinterpret_tensor(buf283, (192, 16), (1, 192), 0); del buf283  # reuse
        buf298 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf290, mul_195, buf296, buf298, 3072, 128, grid=grid(3072), stream=stream0)
        del mul_195
        buf297 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf296, buf297, 192, 16, grid=grid(192), stream=stream0)
        buf299 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf298, buf299, 192, 16, grid=grid(192), stream=stream0)
        buf301 = reinterpret_tensor(buf269, (2048, 384), (384, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (2048, 192), (192, 1), 0), permute_199, out=buf301)
        del permute_199
        buf302 = empty((192, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (192, 2048), (1, 192), 0), view_37, out=buf302)
        del view_37
        buf303 = reinterpret_tensor(buf298, (1, 192, 16), (3072, 1, 192), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf300, buf303, 3072, 128, grid=grid(3072), stream=stream0)
        buf304 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf303, buf304, 192, 16, grid=grid(192), stream=stream0)
        buf305 = reinterpret_tensor(buf301, (32, 64, 384), (24576, 384, 1), 0); del buf301  # reuse
        # Source Nodes: [x_205], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_46.run(buf305, addmm_10, 786432, grid=grid(786432), stream=stream0)
        del addmm_10
        buf306 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (2048, 384), (384, 1), 0), permute_204, out=buf306)
        del permute_204
        buf307 = empty((384, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (384, 2048), (1, 384), 0), view_35, out=buf307)
        del view_35
        buf308 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_47.run(buf305, buf308, 6144, 128, grid=grid(6144), stream=stream0)
        del buf305
        buf309 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_48.run(buf308, buf309, 384, 16, grid=grid(384), stream=stream0)
        del buf308
        buf316 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_49.run(buf316, buf306, primals_122, mul_192, div_15, 2048, 192, grid=grid(2048), stream=stream0)
        del div_15
        del primals_122
        buf312 = reinterpret_tensor(buf303, (192, 16), (1, 192), 0); del buf303  # reuse
        buf314 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf306, mul_192, buf312, buf314, 3072, 128, grid=grid(3072), stream=stream0)
        del mul_192
        buf313 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf312, buf313, 192, 16, grid=grid(192), stream=stream0)
        buf315 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf314, buf315, 192, 16, grid=grid(192), stream=stream0)
        buf317 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (2048, 192), (192, 1), 0), permute_208, out=buf317)
        del permute_208
        buf318 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (192, 2048), (1, 192), 0), view_33, out=buf318)
        del view_33
        buf319 = reinterpret_tensor(buf314, (1, 192, 16), (3072, 1, 192), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf316, buf319, 3072, 128, grid=grid(3072), stream=stream0)
        buf320 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf319, buf320, 192, 16, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf321 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf317, (32, 4, 64, 48), (12288, 48, 192, 1), 0), getitem_72, getitem_73, getitem_74, None, alias_15, getitem_76, getitem_77, getitem_78, 0.0, [True, True, True, False])
        del alias_15
        del buf317
        del getitem_72
        del getitem_73
        del getitem_74
        del getitem_76
        del getitem_77
        del getitem_78
        buf322 = buf321[0]
        buf323 = buf321[1]
        buf324 = buf321[2]
        del buf321
        buf325 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf322, buf323, buf324, buf325, 1179648, grid=grid(1179648), stream=stream0)
        del buf322
        buf326 = reinterpret_tensor(buf324, (2048, 192), (192, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (2048, 576), (576, 1), 0), permute_214, out=buf326)
        del permute_214
        buf327 = empty((576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (576, 2048), (1, 576), 0), view_29, out=buf327)
        del view_29
        buf328 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_52.run(buf325, buf328, 9216, 128, grid=grid(9216), stream=stream0)
        buf329 = empty((1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_53.run(buf328, buf329, 576, 16, grid=grid(576), stream=stream0)
        buf330 = buf188; del buf188  # reuse
        buf331 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_54.run(buf326, primals_116, mul_190, buf330, buf331, 2048, 192, grid=grid(2048), stream=stream0)
        buf332 = reinterpret_tensor(buf319, (192, 16), (1, 192), 0); del buf319  # reuse
        buf334 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf326, mul_190, buf332, buf334, 3072, 128, grid=grid(3072), stream=stream0)
        buf333 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf332, buf333, 192, 16, grid=grid(192), stream=stream0)
        del buf332
        buf335 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf334, buf335, 192, 16, grid=grid(192), stream=stream0)
        del buf334
        buf336 = reinterpret_tensor(buf323, (8, 192, 16, 16), (49152, 256, 16, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_55.run(buf316, div_16, buf326, primals_116, buf330, mul_190, buf331, buf336, 2048, 192, grid=grid(2048, 192), stream=stream0)
        del buf316
        del buf326
        del buf330
        del buf331
        del div_16
        del mul_190
        del primals_116
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf337 = aten.convolution_backward(buf336, mul_189, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf336
        del mul_189
        del primals_115
        buf338 = buf337[0]
        buf339 = buf337[1]
        del buf337
        buf340 = buf179; del buf179  # reuse
        buf341 = empty((128, ), device='cuda', dtype=torch.float32)
        buf342 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_32.run(buf338, mul_539, convolution_23, unsqueeze_238, squeeze_67, buf340, buf341, buf342, 128, 2048, grid=grid(128), stream=stream0)
        buf343 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_33.run(buf343, mul_539, convolution_23, unsqueeze_238, buf341, squeeze_67, buf340, primals_45, 262144, grid=grid(262144), stream=stream0)
        del convolution_23
        del mul_539
        del primals_45
        del squeeze_67
        del unsqueeze_238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf344 = aten.convolution_backward(buf343, add_125, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_125
        del buf343
        del primals_114
        buf345 = buf344[0]
        buf346 = buf344[1]
        del buf344
        buf347 = buf341; del buf341  # reuse
        buf348 = empty((128, ), device='cuda', dtype=torch.float32)
        buf349 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_56.run(buf176, buf345, convolution_22, unsqueeze_250, squeeze_64, buf347, buf348, buf349, 128, 2048, grid=grid(128), stream=stream0)
        buf350 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_57.run(buf350, buf176, convolution_22, unsqueeze_250, buf348, squeeze_64, buf347, primals_43, 262144, grid=grid(262144), stream=stream0)
        del buf176
        del convolution_22
        del primals_43
        del squeeze_64
        del unsqueeze_250
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf351 = aten.convolution_backward(buf350, mul_174, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf350
        del mul_174
        del primals_113
        buf352 = buf351[0]
        buf353 = buf351[1]
        del buf351
        buf354 = empty((384, ), device='cuda', dtype=torch.float32)
        buf355 = empty((384, ), device='cuda', dtype=torch.float32)
        buf356 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_58.run(buf352, mul_560, convolution_21, unsqueeze_262, squeeze_61, buf354, buf355, buf356, 384, 2048, grid=grid(384), stream=stream0)
        buf357 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_59.run(buf357, mul_560, convolution_21, unsqueeze_262, buf355, squeeze_61, buf354, primals_41, 786432, grid=grid(786432), stream=stream0)
        del convolution_21
        del mul_560
        del primals_41
        del squeeze_61
        del unsqueeze_262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf358 = aten.convolution_backward(buf357, mul_166, primals_112, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del buf357
        del mul_166
        del primals_112
        buf359 = buf358[0]
        buf360 = buf358[1]
        del buf358
        buf361 = buf355; del buf355  # reuse
        buf362 = empty((384, ), device='cuda', dtype=torch.float32)
        buf363 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_60.run(buf359, mul_572, convolution_20, unsqueeze_274, squeeze_58, buf361, buf362, buf363, 384, 8192, grid=grid(384), stream=stream0)
        buf364 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_61.run(buf364, mul_572, convolution_20, unsqueeze_274, buf362, squeeze_58, buf361, primals_39, 3145728, grid=grid(3145728), stream=stream0)
        del buf362
        del convolution_20
        del mul_572
        del primals_39
        del squeeze_58
        del unsqueeze_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf365 = aten.convolution_backward(buf364, mul_158, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf364
        del mul_158
        del primals_111
        buf366 = buf365[0]
        buf367 = buf365[1]
        del buf365
        buf368 = empty((96, ), device='cuda', dtype=torch.float32)
        buf369 = empty((96, ), device='cuda', dtype=torch.float32)
        buf370 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_62.run(buf366, mul_584, convolution_19, unsqueeze_286, squeeze_55, buf368, buf369, buf370, 96, 8192, grid=grid(96), stream=stream0)
        buf371 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_63.run(buf371, mul_584, convolution_19, unsqueeze_286, buf369, squeeze_55, buf368, primals_37, 786432, grid=grid(786432), stream=stream0)
        del convolution_19
        del mul_584
        del primals_37
        del squeeze_55
        del unsqueeze_286
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf372 = aten.convolution_backward(buf371, cat, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat
        del primals_110
        buf373 = buf372[0]
        buf374 = buf372[1]
        del buf372
        buf375 = buf369; del buf369  # reuse
        buf376 = empty((96, ), device='cuda', dtype=torch.float32)
        buf377 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_64.run(buf373, mul_596, convolution_18, unsqueeze_298, squeeze_52, buf375, buf376, buf377, 96, 8192, grid=grid(96), stream=stream0)
        buf378 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_65.run(buf373, mul_596, convolution_18, unsqueeze_298, buf376, squeeze_52, buf375, primals_35, buf378, 786432, grid=grid(786432), stream=stream0)
        del convolution_18
        del mul_596
        del primals_35
        del squeeze_52
        del unsqueeze_298
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf379 = aten.convolution_backward(buf378, view_25, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf378
        del primals_109
        del view_25
        buf380 = buf379[0]
        buf381 = buf379[1]
        del buf379
        buf382 = empty_strided((32, 256, 1), (1, 32, 8192), device='cuda', dtype=torch.float32)
        buf383 = empty_strided((32, 256, 1), (256, 1, 8192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_66.run(buf380, primals_107, mul_141, buf382, buf383, 8192, 144, grid=grid(8192), stream=stream0)
        buf384 = reinterpret_tensor(buf325, (32, 256, 144), (36864, 144, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_backward_67.run(div_17, buf380, primals_107, buf382, mul_141, buf383, buf384, 32, 36864, grid=grid(32, 36864), stream=stream0)
        del div_17
        del primals_107
        buf385 = reinterpret_tensor(buf328, (144, 64), (64, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_68.run(buf380, mul_141, buf385, 9216, 128, grid=grid(9216), stream=stream0)
        del mul_141
        buf386 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_69.run(buf385, buf386, 144, 64, grid=grid(144), stream=stream0)
        buf387 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_70.run(buf380, buf387, 144, 8192, grid=grid(144), stream=stream0)
        buf388 = empty((8192, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (8192, 144), (144, 1), 0), permute_227, out=buf388)
        del permute_227
        buf389 = empty((144, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (144, 8192), (1, 144), 0), view_21, out=buf389)
        del view_21
        buf390 = reinterpret_tensor(buf385, (1, 144, 64), (9216, 1, 144), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_71.run(buf384, buf390, 9216, 128, grid=grid(9216), stream=stream0)
        buf391 = empty((1, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_72.run(buf390, buf391, 144, 64, grid=grid(144), stream=stream0)
        buf392 = reinterpret_tensor(buf388, (32, 256, 288), (73728, 288, 1), 0); del buf388  # reuse
        # Source Nodes: [x_145], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_73.run(buf392, addmm_6, 2359296, grid=grid(2359296), stream=stream0)
        del addmm_6
        buf393 = reinterpret_tensor(buf380, (8192, 144), (144, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (8192, 288), (288, 1), 0), permute_232, out=buf393)
        del permute_232
        buf394 = empty((288, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (288, 8192), (1, 288), 0), view_19, out=buf394)
        del view_19
        buf395 = empty_strided((1, 288, 64), (18432, 1, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_74.run(buf392, buf395, 18432, 128, grid=grid(18432), stream=stream0)
        buf396 = empty((1, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_75.run(buf395, buf396, 288, 64, grid=grid(288), stream=stream0)
        buf403 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_76.run(buf403, buf393, primals_101, mul_138, div_18, 8192, 144, grid=grid(8192), stream=stream0)
        del div_18
        del primals_101
        buf399 = reinterpret_tensor(buf390, (144, 64), (1, 144), 0); del buf390  # reuse
        buf401 = empty_strided((144, 64), (1, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_77.run(buf393, mul_138, buf399, buf401, 9216, 128, grid=grid(9216), stream=stream0)
        del mul_138
        buf400 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_72.run(buf399, buf400, 144, 64, grid=grid(144), stream=stream0)
        buf402 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_72.run(buf401, buf402, 144, 64, grid=grid(144), stream=stream0)
        buf404 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (8192, 144), (144, 1), 0), permute_236, out=buf404)
        del permute_236
        buf405 = empty((144, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (144, 8192), (1, 144), 0), view_17, out=buf405)
        del view_17
        buf406 = reinterpret_tensor(buf401, (1, 144, 64), (9216, 1, 144), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_71.run(buf403, buf406, 9216, 128, grid=grid(9216), stream=stream0)
        buf407 = empty((1, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_72.run(buf406, buf407, 144, 64, grid=grid(144), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf408 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf404, (32, 4, 256, 36), (36864, 36, 144, 1), 0), getitem_47, getitem_48, getitem_49, None, alias_16, getitem_51, getitem_52, getitem_53, 0.0, [True, True, True, False])
        del alias_16
        del buf404
        del getitem_47
        del getitem_48
        del getitem_49
        del getitem_51
        del getitem_52
        del getitem_53
        buf409 = buf408[0]
        buf410 = buf408[1]
        buf411 = buf408[2]
        del buf408
        buf412 = empty((32, 256, 3, 4, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_78.run(buf409, buf410, buf411, buf412, 3538944, grid=grid(3538944), stream=stream0)
        del buf409
        del buf410
        buf413 = reinterpret_tensor(buf411, (8192, 144), (144, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (8192, 432), (432, 1), 0), permute_242, out=buf413)
        del permute_242
        buf414 = empty((432, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (432, 8192), (1, 432), 0), view_13, out=buf414)
        del view_13
        buf415 = empty_strided((1, 432, 64), (27648, 1, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_79.run(buf412, buf415, 27648, 128, grid=grid(27648), stream=stream0)
        buf416 = empty((1, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_80.run(buf415, buf416, 432, 64, grid=grid(432), stream=stream0)
        buf423 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_76.run(buf423, buf413, primals_95, mul_136, div_19, 8192, 144, grid=grid(8192), stream=stream0)
        del div_19
        del primals_95
        buf419 = reinterpret_tensor(buf406, (144, 64), (1, 144), 0); del buf406  # reuse
        buf421 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_77.run(buf413, mul_136, buf419, buf421, 9216, 128, grid=grid(9216), stream=stream0)
        del mul_136
        buf420 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_72.run(buf419, buf420, 144, 64, grid=grid(144), stream=stream0)
        buf422 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_72.run(buf421, buf422, 144, 64, grid=grid(144), stream=stream0)
        buf424 = reinterpret_tensor(buf392, (8192, 288), (288, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (8192, 144), (144, 1), 0), permute_246, out=buf424)
        del permute_246
        buf425 = empty((144, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (144, 8192), (1, 144), 0), view_11, out=buf425)
        del view_11
        buf426 = reinterpret_tensor(buf421, (1, 144, 64), (9216, 1, 144), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_71.run(buf423, buf426, 9216, 128, grid=grid(9216), stream=stream0)
        buf427 = empty((1, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_72.run(buf426, buf427, 144, 64, grid=grid(144), stream=stream0)
        buf428 = reinterpret_tensor(buf424, (32, 256, 288), (73728, 288, 1), 0); del buf424  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.add, aten.fill, aten.mul, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_silu_sub_73.run(buf428, addmm_2, 2359296, grid=grid(2359296), stream=stream0)
        del addmm_2
        buf429 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (8192, 288), (288, 1), 0), permute_251, out=buf429)
        del permute_251
        buf430 = empty((288, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (288, 8192), (1, 288), 0), view_9, out=buf430)
        del view_9
        buf431 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_74.run(buf428, buf431, 18432, 128, grid=grid(18432), stream=stream0)
        del buf428
        buf432 = empty((1, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_75.run(buf431, buf432, 288, 64, grid=grid(288), stream=stream0)
        del buf431
        buf439 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_76.run(buf439, buf429, primals_89, mul_133, div_20, 8192, 144, grid=grid(8192), stream=stream0)
        del div_20
        del primals_89
        buf435 = reinterpret_tensor(buf426, (144, 64), (1, 144), 0); del buf426  # reuse
        buf437 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_77.run(buf429, mul_133, buf435, buf437, 9216, 128, grid=grid(9216), stream=stream0)
        del mul_133
        buf436 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_72.run(buf435, buf436, 144, 64, grid=grid(144), stream=stream0)
        buf438 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_72.run(buf437, buf438, 144, 64, grid=grid(144), stream=stream0)
        buf440 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (8192, 144), (144, 1), 0), permute_255, out=buf440)
        del permute_255
        buf441 = empty((144, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (144, 8192), (1, 144), 0), view_7, out=buf441)
        del view_7
        buf442 = reinterpret_tensor(buf437, (1, 144, 64), (9216, 1, 144), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_71.run(buf439, buf442, 9216, 128, grid=grid(9216), stream=stream0)
        buf443 = empty((1, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_72.run(buf442, buf443, 144, 64, grid=grid(144), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf444 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf440, (32, 4, 256, 36), (36864, 36, 144, 1), 0), getitem_36, getitem_37, getitem_38, None, alias_17, getitem_40, getitem_41, getitem_42, 0.0, [True, True, True, False])
        del alias_17
        del buf440
        del getitem_36
        del getitem_37
        del getitem_38
        del getitem_40
        del getitem_41
        del getitem_42
        buf445 = buf444[0]
        buf446 = buf444[1]
        buf447 = buf444[2]
        del buf444
        buf448 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_78.run(buf445, buf446, buf447, buf448, 3538944, grid=grid(3538944), stream=stream0)
        del buf445
        buf449 = reinterpret_tensor(buf447, (8192, 144), (144, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (8192, 432), (432, 1), 0), permute_261, out=buf449)
        del permute_261
        buf450 = empty((432, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (432, 8192), (1, 432), 0), view_3, out=buf450)
        del view_3
        buf451 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_79.run(buf448, buf451, 27648, 128, grid=grid(27648), stream=stream0)
        del buf448
        buf452 = empty((1, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_80.run(buf451, buf452, 432, 64, grid=grid(432), stream=stream0)
        del buf451
        buf453 = buf383; del buf383  # reuse
        buf454 = reinterpret_tensor(buf382, (32, 256, 1), (256, 1, 8192), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_81.run(buf449, primals_83, mul_131, buf453, buf454, 8192, 144, grid=grid(8192), stream=stream0)
        buf455 = reinterpret_tensor(buf442, (144, 64), (1, 144), 0); del buf442  # reuse
        buf457 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_77.run(buf449, mul_131, buf455, buf457, 9216, 128, grid=grid(9216), stream=stream0)
        buf456 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_72.run(buf455, buf456, 144, 64, grid=grid(144), stream=stream0)
        del buf455
        buf458 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_72.run(buf457, buf458, 144, 64, grid=grid(144), stream=stream0)
        del buf457
        buf459 = reinterpret_tensor(buf446, (8, 144, 32, 32), (147456, 1024, 32, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_82.run(buf439, div_21, buf449, primals_83, buf453, mul_131, buf454, buf459, 8192, 144, grid=grid(8192, 144), stream=stream0)
        del buf439
        del buf449
        del buf453
        del buf454
        del div_21
        del mul_131
        del primals_83
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf460 = aten.convolution_backward(buf459, mul_130, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf459
        del mul_130
        del primals_82
        buf461 = buf460[0]
        buf462 = buf460[1]
        del buf460
        buf463 = buf376; del buf376  # reuse
        buf464 = empty((96, ), device='cuda', dtype=torch.float32)
        buf465 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_62.run(buf461, mul_649, convolution_16, unsqueeze_310, squeeze_49, buf463, buf464, buf465, 96, 8192, grid=grid(96), stream=stream0)
        buf466 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_63.run(buf466, mul_649, convolution_16, unsqueeze_310, buf464, squeeze_49, buf463, primals_33, 786432, grid=grid(786432), stream=stream0)
        del convolution_16
        del mul_649
        del primals_33
        del squeeze_49
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf467 = aten.convolution_backward(buf466, add_81, primals_81, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_81
        del buf466
        del primals_81
        buf468 = buf467[0]
        buf469 = buf467[1]
        del buf467
        buf470 = buf464; del buf464  # reuse
        buf471 = empty((96, ), device='cuda', dtype=torch.float32)
        buf472 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_83.run(buf373, buf468, convolution_15, unsqueeze_322, squeeze_46, buf470, buf471, buf472, 96, 8192, grid=grid(96), stream=stream0)
        buf473 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_84.run(buf473, buf373, convolution_15, unsqueeze_322, buf471, squeeze_46, buf470, primals_31, 786432, grid=grid(786432), stream=stream0)
        del buf373
        del buf471
        del convolution_15
        del primals_31
        del squeeze_46
        del unsqueeze_322
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf474 = aten.convolution_backward(buf473, mul_115, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf473
        del mul_115
        del primals_80
        buf475 = buf474[0]
        buf476 = buf474[1]
        del buf474
        buf477 = empty((256, ), device='cuda', dtype=torch.float32)
        buf478 = empty((256, ), device='cuda', dtype=torch.float32)
        buf479 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_85.run(buf475, mul_670, convolution_14, unsqueeze_334, squeeze_43, buf477, buf478, buf479, 256, 8192, grid=grid(256), stream=stream0)
        buf480 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_86.run(buf480, mul_670, convolution_14, unsqueeze_334, buf478, squeeze_43, buf477, primals_29, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_14
        del mul_670
        del primals_29
        del squeeze_43
        del unsqueeze_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf481 = aten.convolution_backward(buf480, mul_107, primals_79, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False])
        del mul_107
        del primals_79
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        buf484 = buf478; del buf478  # reuse
        buf485 = empty((256, ), device='cuda', dtype=torch.float32)
        buf486 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_87.run(buf482, mul_682, convolution_13, unsqueeze_346, squeeze_40, buf484, buf485, buf486, 256, 32768, grid=grid(256), stream=stream0)
        buf487 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_88.run(buf487, mul_682, convolution_13, unsqueeze_346, buf485, squeeze_40, buf484, primals_27, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_13
        del mul_682
        del primals_27
        del squeeze_40
        del unsqueeze_346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf488 = aten.convolution_backward(buf487, add_66, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_66
        del buf487
        del primals_78
        buf489 = buf488[0]
        buf490 = buf488[1]
        del buf488
        buf491 = reinterpret_tensor(buf485, (64, 4), (1, 64), 0); del buf485  # reuse
        buf493 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_89.run(buf489, convolution_12, unsqueeze_358, buf491, buf493, 256, 8192, grid=grid(256), stream=stream0)
        buf492 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_90.run(buf491, buf492, 64, 4, grid=grid(64), stream=stream0)
        buf494 = empty((64, ), device='cuda', dtype=torch.float32)
        buf495 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_91.run(buf493, squeeze_37, buf494, buf495, 64, 4, grid=grid(64), stream=stream0)
        buf496 = reinterpret_tensor(buf480, (8, 64, 64, 64), (262144, 4096, 64, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_92.run(buf489, convolution_12, unsqueeze_358, buf494, squeeze_37, buf492, primals_25, buf496, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_12
        del primals_25
        del squeeze_37
        del unsqueeze_358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf497 = aten.convolution_backward(buf496, mul_92, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_92
        del primals_77
        buf498 = buf497[0]
        buf499 = buf497[1]
        del buf497
        buf500 = reinterpret_tensor(buf493, (256, ), (1, ), 0); del buf493  # reuse
        buf501 = reinterpret_tensor(buf491, (256, ), (1, ), 0); del buf491  # reuse
        buf502 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_87.run(buf498, mul_703, convolution_11, unsqueeze_370, squeeze_34, buf500, buf501, buf502, 256, 32768, grid=grid(256), stream=stream0)
        buf503 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_88.run(buf503, mul_703, convolution_11, unsqueeze_370, buf501, squeeze_34, buf500, primals_23, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_11
        del mul_703
        del primals_23
        del squeeze_34
        del unsqueeze_370
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf504 = aten.convolution_backward(buf503, mul_84, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False])
        del buf503
        del mul_84
        del primals_76
        buf505 = buf504[0]
        buf506 = buf504[1]
        del buf504
        buf507 = buf501; del buf501  # reuse
        buf508 = empty((256, ), device='cuda', dtype=torch.float32)
        buf509 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_87.run(buf505, mul_715, convolution_10, unsqueeze_382, squeeze_31, buf507, buf508, buf509, 256, 32768, grid=grid(256), stream=stream0)
        buf510 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_88.run(buf510, mul_715, convolution_10, unsqueeze_382, buf508, squeeze_31, buf507, primals_21, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_10
        del mul_715
        del primals_21
        del squeeze_31
        del unsqueeze_382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf511 = aten.convolution_backward(buf510, add_50, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_50
        del buf510
        del primals_75
        buf512 = buf511[0]
        buf513 = buf511[1]
        del buf511
        buf514 = reinterpret_tensor(buf508, (64, 4), (1, 64), 0); del buf508  # reuse
        buf516 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_93.run(buf489, buf512, convolution_9, unsqueeze_394, buf514, buf516, 256, 8192, grid=grid(256), stream=stream0)
        buf515 = buf494; del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_90.run(buf514, buf515, 64, 4, grid=grid(64), stream=stream0)
        buf517 = empty((64, ), device='cuda', dtype=torch.float32)
        buf518 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_91.run(buf516, squeeze_28, buf517, buf518, 64, 4, grid=grid(64), stream=stream0)
        buf519 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_94.run(buf489, buf512, convolution_9, unsqueeze_394, buf517, squeeze_28, buf515, primals_19, buf519, 2097152, grid=grid(2097152), stream=stream0)
        del convolution_9
        del primals_19
        del squeeze_28
        del unsqueeze_394
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf520 = aten.convolution_backward(buf519, mul_69, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf519
        del mul_69
        del primals_74
        buf521 = buf520[0]
        buf522 = buf520[1]
        del buf520
        buf523 = reinterpret_tensor(buf516, (256, ), (1, ), 0); del buf516  # reuse
        buf524 = reinterpret_tensor(buf514, (256, ), (1, ), 0); del buf514  # reuse
        buf525 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_87.run(buf521, mul_736, convolution_8, unsqueeze_406, squeeze_25, buf523, buf524, buf525, 256, 32768, grid=grid(256), stream=stream0)
        buf526 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_88.run(buf526, mul_736, convolution_8, unsqueeze_406, buf524, squeeze_25, buf523, primals_17, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_8
        del mul_736
        del primals_17
        del squeeze_25
        del unsqueeze_406
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf527 = aten.convolution_backward(buf526, mul_61, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False])
        del buf526
        del mul_61
        del primals_73
        buf528 = buf527[0]
        buf529 = buf527[1]
        del buf527
        buf530 = buf524; del buf524  # reuse
        buf531 = empty((256, ), device='cuda', dtype=torch.float32)
        buf532 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_87.run(buf528, mul_748, convolution_7, unsqueeze_418, squeeze_22, buf530, buf531, buf532, 256, 32768, grid=grid(256), stream=stream0)
        buf533 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_88.run(buf533, mul_748, convolution_7, unsqueeze_418, buf531, squeeze_22, buf530, primals_15, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_7
        del mul_748
        del primals_15
        del squeeze_22
        del unsqueeze_418
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf534 = aten.convolution_backward(buf533, add_34, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_34
        del buf533
        del primals_72
        buf535 = buf534[0]
        buf536 = buf534[1]
        del buf534
        buf537 = reinterpret_tensor(buf531, (64, 4), (1, 64), 0); del buf531  # reuse
        buf539 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_95.run(buf489, buf512, buf535, convolution_6, unsqueeze_430, buf537, buf539, 256, 8192, grid=grid(256), stream=stream0)
        buf538 = buf517; del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_90.run(buf537, buf538, 64, 4, grid=grid(64), stream=stream0)
        buf540 = empty((64, ), device='cuda', dtype=torch.float32)
        buf542 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_91.run(buf539, squeeze_19, buf540, buf542, 64, 4, grid=grid(64), stream=stream0)
        buf541 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_96.run(buf541, buf512, buf535, convolution_6, unsqueeze_430, buf540, squeeze_19, buf538, primals_13, 2097152, grid=grid(2097152), stream=stream0)
        del buf512
        del buf535
        del convolution_6
        del primals_13
        del squeeze_19
        del unsqueeze_430
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf543 = aten.convolution_backward(buf541, mul_46, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf541
        del mul_46
        del primals_71
        buf544 = buf543[0]
        buf545 = buf543[1]
        del buf543
        buf546 = reinterpret_tensor(buf165, (128, 4), (1, 128), 0); del buf165  # reuse
        buf548 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_97.run(buf544, mul_769, convolution_5, unsqueeze_442, buf546, buf548, 512, 8192, grid=grid(512), stream=stream0)
        buf547 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_98.run(buf546, buf547, 128, 4, grid=grid(128), stream=stream0)
        buf549 = empty((128, ), device='cuda', dtype=torch.float32)
        buf550 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_99.run(buf548, squeeze_16, buf549, buf550, 128, 4, grid=grid(128), stream=stream0)
        buf551 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_100.run(buf551, mul_769, convolution_5, unsqueeze_442, buf549, squeeze_16, buf547, primals_11, 4194304, grid=grid(4194304), stream=stream0)
        del convolution_5
        del mul_769
        del primals_11
        del squeeze_16
        del unsqueeze_442
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf552 = aten.convolution_backward(buf551, mul_38, primals_70, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del buf551
        del mul_38
        del primals_70
        buf553 = buf552[0]
        buf554 = buf552[1]
        del buf552
        buf555 = buf548; del buf548  # reuse
        buf557 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_101.run(buf553, mul_781, convolution_4, unsqueeze_454, buf555, buf557, 512, 32768, grid=grid(512), stream=stream0)
        buf556 = buf549; del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_98.run(buf555, buf556, 128, 4, grid=grid(128), stream=stream0)
        buf558 = empty((128, ), device='cuda', dtype=torch.float32)
        buf559 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_99.run(buf557, squeeze_13, buf558, buf559, 128, 4, grid=grid(128), stream=stream0)
        buf560 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_102.run(buf560, mul_781, convolution_4, unsqueeze_454, buf558, squeeze_13, buf556, primals_9, 16777216, grid=grid(16777216), stream=stream0)
        del buf558
        del convolution_4
        del mul_781
        del primals_9
        del squeeze_13
        del unsqueeze_454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf561 = aten.convolution_backward(buf560, add_19, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_19
        del buf560
        del primals_69
        buf562 = buf561[0]
        buf563 = buf561[1]
        del buf561
        buf564 = reinterpret_tensor(buf557, (32, 16), (16, 1), 0); del buf557  # reuse
        buf566 = reinterpret_tensor(buf555, (32, 16), (16, 1), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_103.run(buf562, convolution_3, unsqueeze_466, buf564, buf566, 512, 8192, grid=grid(512), stream=stream0)
        buf565 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_104.run(buf564, buf565, 32, 16, grid=grid(32), stream=stream0)
        del buf564
        buf567 = empty((32, ), device='cuda', dtype=torch.float32)
        buf568 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_105.run(buf566, squeeze_10, buf567, buf568, 32, 16, grid=grid(32), stream=stream0)
        del buf566
        buf569 = buf562; del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_106.run(buf569, convolution_3, unsqueeze_466, buf567, squeeze_10, buf565, primals_7, 4194304, grid=grid(4194304), stream=stream0)
        del buf567
        del convolution_3
        del primals_7
        del squeeze_10
        del unsqueeze_466
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf570 = aten.convolution_backward(buf569, mul_23, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf569
        del mul_23
        del primals_68
        buf571 = buf570[0]
        buf572 = buf570[1]
        del buf570
        buf573 = reinterpret_tensor(buf26, (64, 16), (16, 1), 0); del buf26  # reuse
        buf575 = empty((64, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_107.run(buf571, mul_802, convolution_2, unsqueeze_478, buf573, buf575, 1024, 8192, grid=grid(1024), stream=stream0)
        buf574 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_108.run(buf573, buf574, 64, 16, grid=grid(64), stream=stream0)
        buf576 = empty((64, ), device='cuda', dtype=torch.float32)
        buf577 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_109.run(buf575, squeeze_7, buf576, buf577, 64, 16, grid=grid(64), stream=stream0)
        buf578 = buf571; del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_110.run(buf578, mul_802, convolution_2, unsqueeze_478, buf576, squeeze_7, buf574, primals_5, 8388608, grid=grid(8388608), stream=stream0)
        del convolution_2
        del mul_802
        del primals_5
        del squeeze_7
        del unsqueeze_478
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf579 = aten.convolution_backward(buf578, mul_15, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del buf578
        del mul_15
        del primals_67
        buf580 = buf579[0]
        buf581 = buf579[1]
        del buf579
        buf582 = buf575; del buf575  # reuse
        buf584 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_107.run(buf580, mul_814, convolution_1, unsqueeze_490, buf582, buf584, 1024, 8192, grid=grid(1024), stream=stream0)
        buf583 = buf576; del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_108.run(buf582, buf583, 64, 16, grid=grid(64), stream=stream0)
        del buf582
        buf585 = empty((64, ), device='cuda', dtype=torch.float32)
        buf586 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_109.run(buf584, squeeze_4, buf585, buf586, 64, 16, grid=grid(64), stream=stream0)
        del buf584
        buf587 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_110.run(buf587, mul_814, convolution_1, unsqueeze_490, buf585, squeeze_4, buf583, primals_3, 8388608, grid=grid(8388608), stream=stream0)
        del buf585
        del convolution_1
        del mul_814
        del primals_3
        del squeeze_4
        del unsqueeze_490
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf588 = aten.convolution_backward(buf587, mul_7, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf587
        del mul_7
        del primals_66
        buf589 = buf588[0]
        buf590 = buf588[1]
        del buf588
        buf591 = reinterpret_tensor(buf539, (16, 16), (16, 1), 0); del buf539  # reuse
        buf593 = reinterpret_tensor(buf537, (16, 16), (16, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_111.run(buf589, mul_826, convolution, unsqueeze_502, buf591, buf593, 256, 8192, grid=grid(256), stream=stream0)
        buf592 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_112.run(buf591, buf592, 16, 16, grid=grid(16), stream=stream0)
        del buf591
        buf594 = empty((16, ), device='cuda', dtype=torch.float32)
        buf595 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_113.run(buf593, squeeze_1, buf594, buf595, 16, 16, grid=grid(16), stream=stream0)
        del buf593
        buf596 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_114.run(buf596, mul_826, convolution, unsqueeze_502, buf594, squeeze_1, buf592, primals_1, 2097152, grid=grid(2097152), stream=stream0)
        del buf594
        del convolution
        del mul_826
        del primals_1
        del squeeze_1
        del unsqueeze_502
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf597 = aten.convolution_backward(buf596, primals_312, primals_65, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf596
        del primals_312
        del primals_65
        buf598 = buf597[1]
        return (buf595, buf592, buf586, buf583, buf577, buf574, buf568, buf565, buf559, buf556, buf550, buf547, buf542, buf538, buf532, buf530, buf525, buf523, buf518, buf515, buf509, buf507, buf502, buf500, buf495, buf492, buf486, buf484, buf479, buf477, buf472, buf470, buf465, buf463, buf377, buf375, buf370, buf368, buf363, buf361, buf356, buf354, buf349, buf347, buf342, buf340, buf180, buf178, buf173, buf171, buf166, buf164, buf159, buf157, buf152, buf150, buf145, buf143, buf19, buf17, buf12, buf10, buf5, buf3, buf598, buf590, buf581, buf572, buf563, buf554, buf545, buf536, buf529, buf522, buf513, buf506, buf499, buf490, buf483, buf476, buf469, buf462, buf456, buf458, reinterpret_tensor(buf450, (432, 144), (144, 1), 0), reinterpret_tensor(buf452, (432, ), (1, ), 0), reinterpret_tensor(buf441, (144, 144), (144, 1), 0), reinterpret_tensor(buf443, (144, ), (1, ), 0), buf436, buf438, reinterpret_tensor(buf430, (288, 144), (144, 1), 0), reinterpret_tensor(buf432, (288, ), (1, ), 0), reinterpret_tensor(buf425, (144, 288), (288, 1), 0), reinterpret_tensor(buf427, (144, ), (1, ), 0), buf420, buf422, reinterpret_tensor(buf414, (432, 144), (144, 1), 0), reinterpret_tensor(buf416, (432, ), (1, ), 0), reinterpret_tensor(buf405, (144, 144), (144, 1), 0), reinterpret_tensor(buf407, (144, ), (1, ), 0), buf400, buf402, reinterpret_tensor(buf394, (288, 144), (144, 1), 0), reinterpret_tensor(buf396, (288, ), (1, ), 0), reinterpret_tensor(buf389, (144, 288), (288, 1), 0), reinterpret_tensor(buf391, (144, ), (1, ), 0), buf386, buf387, buf381, buf374, buf367, buf360, buf353, buf346, buf339, buf333, buf335, reinterpret_tensor(buf327, (576, 192), (192, 1), 0), reinterpret_tensor(buf329, (576, ), (1, ), 0), reinterpret_tensor(buf318, (192, 192), (192, 1), 0), reinterpret_tensor(buf320, (192, ), (1, ), 0), buf313, buf315, reinterpret_tensor(buf307, (384, 192), (192, 1), 0), reinterpret_tensor(buf309, (384, ), (1, ), 0), reinterpret_tensor(buf302, (192, 384), (384, 1), 0), reinterpret_tensor(buf304, (192, ), (1, ), 0), buf297, buf299, reinterpret_tensor(buf291, (576, 192), (192, 1), 0), reinterpret_tensor(buf293, (576, ), (1, ), 0), reinterpret_tensor(buf282, (192, 192), (192, 1), 0), reinterpret_tensor(buf284, (192, ), (1, ), 0), buf277, buf279, reinterpret_tensor(buf271, (384, 192), (192, 1), 0), reinterpret_tensor(buf273, (384, ), (1, ), 0), reinterpret_tensor(buf266, (192, 384), (384, 1), 0), reinterpret_tensor(buf268, (192, ), (1, ), 0), buf261, buf263, reinterpret_tensor(buf255, (576, 192), (192, 1), 0), reinterpret_tensor(buf257, (576, ), (1, ), 0), reinterpret_tensor(buf246, (192, 192), (192, 1), 0), reinterpret_tensor(buf248, (192, ), (1, ), 0), buf241, buf243, reinterpret_tensor(buf235, (384, 192), (192, 1), 0), reinterpret_tensor(buf237, (384, ), (1, ), 0), reinterpret_tensor(buf230, (192, 384), (384, 1), 0), reinterpret_tensor(buf232, (192, ), (1, ), 0), buf225, buf227, reinterpret_tensor(buf219, (576, 192), (192, 1), 0), reinterpret_tensor(buf221, (576, ), (1, ), 0), reinterpret_tensor(buf210, (192, 192), (192, 1), 0), reinterpret_tensor(buf212, (192, ), (1, ), 0), buf205, buf207, reinterpret_tensor(buf199, (384, 192), (192, 1), 0), reinterpret_tensor(buf201, (384, ), (1, ), 0), reinterpret_tensor(buf194, (192, 384), (384, 1), 0), reinterpret_tensor(buf196, (192, ), (1, ), 0), buf191, buf192, buf184, buf177, buf170, buf163, buf156, buf149, buf142, buf136, buf138, reinterpret_tensor(buf130, (720, 240), (240, 1), 0), reinterpret_tensor(buf132, (720, ), (1, ), 0), reinterpret_tensor(buf121, (240, 240), (240, 1), 0), reinterpret_tensor(buf123, (240, ), (1, ), 0), buf116, buf118, reinterpret_tensor(buf110, (480, 240), (240, 1), 0), reinterpret_tensor(buf112, (480, ), (1, ), 0), reinterpret_tensor(buf105, (240, 480), (480, 1), 0), reinterpret_tensor(buf107, (240, ), (1, ), 0), buf100, buf102, reinterpret_tensor(buf94, (720, 240), (240, 1), 0), reinterpret_tensor(buf96, (720, ), (1, ), 0), reinterpret_tensor(buf85, (240, 240), (240, 1), 0), reinterpret_tensor(buf87, (240, ), (1, ), 0), buf80, buf82, reinterpret_tensor(buf74, (480, 240), (240, 1), 0), reinterpret_tensor(buf76, (480, ), (1, ), 0), reinterpret_tensor(buf69, (240, 480), (480, 1), 0), reinterpret_tensor(buf71, (240, ), (1, ), 0), buf64, buf66, reinterpret_tensor(buf58, (720, 240), (240, 1), 0), reinterpret_tensor(buf60, (720, ), (1, ), 0), reinterpret_tensor(buf49, (240, 240), (240, 1), 0), reinterpret_tensor(buf51, (240, ), (1, ), 0), buf44, buf46, reinterpret_tensor(buf38, (480, 240), (240, 1), 0), reinterpret_tensor(buf40, (480, ), (1, ), 0), reinterpret_tensor(buf33, (240, 480), (480, 1), 0), reinterpret_tensor(buf35, (240, ), (1, ), 0), buf30, buf31, buf23, buf16, buf9, reinterpret_tensor(buf1, (1000, 640), (640, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((144, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((96, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((96, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((160, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((160, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((160, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 16, 128, 128), (262144, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 16, 128, 128), (262144, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_19 = rand_strided((8, 32, 128, 128), (524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_38 = rand_strided((8, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_46 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_34 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_61 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_69 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_50 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_92 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_66 = rand_strided((8, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_107 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_115 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_81 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_130 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    mul_131 = rand_strided((32, 256, 144), (36864, 144, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((8192, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    getitem_36 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cuda:0', dtype=torch.float32)
    getitem_38 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((32, 4, 256), (1024, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_42 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_7 = rand_strided((8192, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((32, 256, 144), (36864, 144, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((8192, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((8192, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((8192, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    mul_136 = rand_strided((32, 256, 144), (36864, 144, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((8192, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((32, 4, 256), (1024, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_52 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_53 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_17 = rand_strided((8192, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    mul_138 = rand_strided((32, 256, 144), (36864, 144, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((8192, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((8192, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((8192, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    mul_141 = rand_strided((32, 256, 144), (36864, 144, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((8, 144, 32, 32), (147456, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 192, 32, 32), (196608, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_158 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 384, 32, 32), (393216, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_166 = rand_strided((8, 384, 32, 32), (393216, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 384, 16, 16), (98304, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_174 = rand_strided((8, 384, 16, 16), (98304, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_125 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_189 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    mul_190 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_74 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_76 = rand_strided((32, 4, 64), (256, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_78 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_33 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((2048, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((2048, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_195 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((32, 4, 64), (256, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_88 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_89 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_43 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_197 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((2048, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((2048, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_200 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_96 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_98 = rand_strided((32, 4, 64), (256, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_99 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_100 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_53 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_202 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((2048, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((2048, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_205 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_106 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cuda:0', dtype=torch.float32)
    getitem_109 = rand_strided((32, 4, 64), (256, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_111 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_63 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_207 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((2048, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((2048, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((2048, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_210 = rand_strided((32, 64, 192), (12288, 192, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((8, 192, 16, 16), (49152, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 256, 16, 16), (65536, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_227 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_235 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_243 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_181 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_258 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    mul_259 = rand_strided((32, 16, 240), (3840, 240, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    getitem_130 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_131 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_132 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((32, 4, 32), (128, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_79 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    mul_261 = rand_strided((32, 16, 240), (3840, 240, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((512, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((512, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    mul_264 = rand_strided((32, 16, 240), (3840, 240, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_142 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_143 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_145 = rand_strided((32, 4, 32), (128, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_146 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_147 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_89 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    mul_266 = rand_strided((32, 16, 240), (3840, 240, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((512, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((512, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    mul_269 = rand_strided((32, 16, 240), (3840, 240, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    getitem_152 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_153 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_154 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cuda:0', dtype=torch.float32)
    getitem_156 = rand_strided((32, 4, 32), (128, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_157 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_158 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_99 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    mul_271 = rand_strided((32, 16, 240), (3840, 240, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((512, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((512, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    mul_274 = rand_strided((32, 16, 240), (3840, 240, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((8, 240, 8, 8), (15360, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 320, 8, 8), (20480, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_291 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_64 = rand_strided((8, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    permute_67 = rand_strided((1000, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    mul_301 = rand_strided((8, 640, 8, 8), (40960, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_313 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_142 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_325 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((32, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_76 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    permute_81 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((32, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_85 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    alias_9 = rand_strided((32, 4, 16, 60), (3840, 60, 240, 1), device='cuda:0', dtype=torch.float32)
    permute_91 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((32, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((32, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_104 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    alias_10 = rand_strided((32, 4, 16, 60), (3840, 60, 240, 1), device='cuda:0', dtype=torch.float32)
    permute_110 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((32, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_114 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    permute_119 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((32, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    alias_11 = rand_strided((32, 4, 16, 60), (3840, 60, 240, 1), device='cuda:0', dtype=torch.float32)
    permute_129 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((32, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_395 = rand_strided((8, 160, 8, 8), (10240, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_166 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_416 = rand_strided((8, 512, 8, 8), (32768, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_428 = rand_strided((8, 512, 16, 16), (131072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_440 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_452 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((32, 4, 64, 48), (12288, 48, 192, 1), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_161 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((32, 4, 64, 48), (12288, 48, 192, 1), device='cuda:0', dtype=torch.float32)
    permute_176 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((32, 4, 64, 48), (12288, 48, 192, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((32, 4, 64, 48), (12288, 48, 192, 1), device='cuda:0', dtype=torch.float32)
    permute_214 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_539 = rand_strided((8, 128, 16, 16), (32768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_560 = rand_strided((8, 384, 16, 16), (98304, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_572 = rand_strided((8, 384, 32, 32), (393216, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_584 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_596 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((144, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((288, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((144, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((32, 4, 256, 36), (36864, 36, 144, 1), device='cuda:0', dtype=torch.float32)
    permute_242 = rand_strided((432, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_246 = rand_strided((144, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((288, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((144, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((32, 4, 256, 36), (36864, 36, 144, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((432, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_649 = rand_strided((8, 96, 32, 32), (98304, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_670 = rand_strided((8, 256, 32, 32), (262144, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_682 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_703 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_715 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_736 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_748 = rand_strided((8, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_769 = rand_strided((8, 128, 64, 64), (524288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_781 = rand_strided((8, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_802 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_814 = rand_strided((8, 64, 128, 128), (1048576, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_826 = rand_strided((8, 16, 128, 128), (262144, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_89, primals_95, primals_101, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_152, primals_158, primals_164, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_211, primals_212, primals_213, primals_312, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, add_19, convolution_4, squeeze_13, mul_38, convolution_5, squeeze_16, mul_46, convolution_6, squeeze_19, add_34, convolution_7, squeeze_22, mul_61, convolution_8, squeeze_25, mul_69, convolution_9, squeeze_28, add_50, convolution_10, squeeze_31, mul_84, convolution_11, squeeze_34, mul_92, convolution_12, squeeze_37, add_66, convolution_13, squeeze_40, mul_107, convolution_14, squeeze_43, mul_115, convolution_15, squeeze_46, add_81, convolution_16, squeeze_49, mul_130, mul_131, view_3, getitem_36, getitem_37, getitem_38, getitem_40, getitem_41, getitem_42, view_7, mul_133, view_9, addmm_2, view_11, mul_136, view_13, getitem_47, getitem_48, getitem_49, getitem_51, getitem_52, getitem_53, view_17, mul_138, view_19, addmm_6, view_21, mul_141, view_25, convolution_18, squeeze_52, cat, convolution_19, squeeze_55, mul_158, convolution_20, squeeze_58, mul_166, convolution_21, squeeze_61, mul_174, convolution_22, squeeze_64, add_125, convolution_23, squeeze_67, mul_189, mul_190, view_29, getitem_72, getitem_73, getitem_74, getitem_76, getitem_77, getitem_78, view_33, mul_192, view_35, addmm_10, view_37, mul_195, view_39, getitem_83, getitem_84, getitem_85, getitem_87, getitem_88, getitem_89, view_43, mul_197, view_45, addmm_14, view_47, mul_200, view_49, getitem_94, getitem_95, getitem_96, getitem_98, getitem_99, getitem_100, view_53, mul_202, view_55, addmm_18, view_57, mul_205, view_59, getitem_105, getitem_106, getitem_107, getitem_109, getitem_110, getitem_111, view_63, mul_207, view_65, addmm_22, view_67, mul_210, view_71, convolution_25, squeeze_70, cat_1, convolution_26, squeeze_73, mul_227, convolution_27, squeeze_76, mul_235, convolution_28, squeeze_79, mul_243, convolution_29, squeeze_82, add_181, convolution_30, squeeze_85, mul_258, mul_259, view_75, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, view_79, mul_261, view_81, addmm_26, view_83, mul_264, view_85, getitem_141, getitem_142, getitem_143, getitem_145, getitem_146, getitem_147, view_89, mul_266, view_91, addmm_30, view_93, mul_269, view_95, getitem_152, getitem_153, getitem_154, getitem_156, getitem_157, getitem_158, view_99, mul_271, view_101, addmm_34, view_103, mul_274, view_107, convolution_32, squeeze_88, cat_2, convolution_33, squeeze_91, mul_291, convolution_34, squeeze_94, clone_64, permute_67, mul_301, unsqueeze_130, mul_313, unsqueeze_142, mul_325, unsqueeze_154, div_1, permute_76, permute_81, div_2, permute_85, alias_9, permute_91, div_3, permute_95, permute_100, div_4, permute_104, alias_10, permute_110, div_5, permute_114, permute_119, div_6, permute_123, alias_11, permute_129, div_7, mul_395, unsqueeze_166, unsqueeze_178, mul_416, unsqueeze_190, mul_428, unsqueeze_202, mul_440, unsqueeze_214, mul_452, unsqueeze_226, div_8, permute_142, permute_147, div_9, permute_151, alias_12, permute_157, div_10, permute_161, permute_166, div_11, permute_170, alias_13, permute_176, div_12, permute_180, permute_185, div_13, permute_189, alias_14, permute_195, div_14, permute_199, permute_204, div_15, permute_208, alias_15, permute_214, div_16, mul_539, unsqueeze_238, unsqueeze_250, mul_560, unsqueeze_262, mul_572, unsqueeze_274, mul_584, unsqueeze_286, mul_596, unsqueeze_298, div_17, permute_227, permute_232, div_18, permute_236, alias_16, permute_242, div_19, permute_246, permute_251, div_20, permute_255, alias_17, permute_261, div_21, mul_649, unsqueeze_310, unsqueeze_322, mul_670, unsqueeze_334, mul_682, unsqueeze_346, unsqueeze_358, mul_703, unsqueeze_370, mul_715, unsqueeze_382, unsqueeze_394, mul_736, unsqueeze_406, mul_748, unsqueeze_418, unsqueeze_430, mul_769, unsqueeze_442, mul_781, unsqueeze_454, unsqueeze_466, mul_802, unsqueeze_478, mul_814, unsqueeze_490, mul_826, unsqueeze_502, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilevit_s', benchmark_compiled_module)
