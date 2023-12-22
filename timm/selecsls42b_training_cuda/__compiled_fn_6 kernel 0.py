
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


# kernel path: /tmp/torchinductor_youkaichao/my/cmyv7c27a5wkdr6stsii3w52qobqdzqj32f2irbfwyjc4qqjevth.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 16
    r2 = (rindex // 16)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0) + (16384*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (16*x0) + (16384*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 16.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/ccltz7pbwiazinmvrngxisocqdkmpqstls4le7eezcxrzrs54wn4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 16)
    x1 = (xindex // 16) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = 16.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.0078125
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnny2o6vhmlzmhxyfj63pl4zzszx3ok6dzkfdc4nz5wa6halcmfs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 16
    r2 = (rindex // 16)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0) + (20480*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (16*x0) + (20480*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (16*x0) + (20480*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmij6dudwb6gbqrot33a4vox7xk2vsxuqqm673tesi7mz2h2bu6m.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16) % 1280
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
    tmp9 = 0.0078125
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


# kernel path: /tmp/torchinductor_youkaichao/r3/cr356wcfe24n4ddigudzmkszawthcgjdra3rbojofwl2juhfsdwo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbgr6cklhdgivnpew2fq7k27wmp7mwgkpeqmjwlv53p7duqhytq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
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
    tmp9 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/d7/cd7n2aegsknngzrua4zgnq5lqgvwzzsq2ez4akdf4uzbmvhgdmro.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 960
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (47040*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (47040*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (49*x0) + (47040*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnl24suonaiex34ckh6s5uhk3avjo63pw64rq2m3gfqqicjcbz7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 960
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlbluh4e6opcqj5dvplkmd4fd5ll5dxhyb2xdb24s66kk7tfyc3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (94080*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (94080*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (94080*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hg/chgqp7kghlfkc5hmmxmtxhrhnb4my77vrnk2auqdeu7s7pa2p6i5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 480
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciueo7b3pg7oqz6b2qeacif4nn423invlgnpbdanbf7osqpfvu7p.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (89376 + r1 + (196*x0) + (178752*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tmp12 * tmp14
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3m7oaf7exutctwd6dtfeddcmnbytqn2lecb2y7e6vw7wuldfti.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 29792)
    x4 = xindex % 29792
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (89376 + x4 + (178752*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.0006377551020408163
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf36jk7qaz4a7hdxvwwllcaxpt37jbxmlihcgh26ny6khotnvaua.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 304
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/o5/co5edviwe6v6lixkit4u3qsdvfnz3sdqqtizfo67by2mw7n2hrgp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 304
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x3), xmask)
    tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hw/chwiqhpsqgtb6sdhmqu4po2bup7ngzmtn7ixyvkrfs7a7ozttlyj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (59584 + r1 + (196*x0) + (178752*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/n6/cn65u3deihnuqvebd4gmcisi3w7nqo6lqcyjvdns67qxd5patpia.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 29792)
    x4 = xindex % 29792
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (59584 + x4 + (178752*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ye/cyeqnkbtbc6d5zh5kizaii4fzo6lsypce3rp3pytdmdokakgixv3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 304
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (178752*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5i/c5iwr4obbhmwdqepyfvt73lyht64ywweezvfe4scglb6zuvgbj6m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 59584)
    x4 = xindex % 59584
    x1 = (xindex // 196) % 304
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x4 + (178752*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sx/csxn37hihv74jcqme67vnqhh6coejahpa27ixrknvz3uiipl3ylo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 304
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (119168 + r1 + (196*x0) + (178752*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wn/cwnyy244537qrrac35tgbiuwxu76uoy2rid77rne4dj4ybc53aue.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 59584)
    x4 = xindex % 59584
    x1 = (xindex // 196) % 304
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (119168 + x4 + (178752*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kitfkmsbgnbfha4n6z7nj3kyesunmqs53sza5aljj55o2t73q7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (89376 + r1 + (196*x0) + (119168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tmp12 * tmp14
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd25whna4xbd4j5mw4afldp6bqgv42qa3h7gztvdbnyjo32dxijf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 29792)
    x4 = xindex % 29792
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (89376 + x4 + (119168*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.0006377551020408163
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cix4oqljidub6nqrd4j3dnnpv5j4qb5isjnlxtxqn5p2hq6oelgo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (59584 + r1 + (196*x0) + (119168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yu/cyujpt4zdvsutg4ohpd4me7lslmftnhpj2k5lluzkzyl7utdvyca.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 29792)
    x4 = xindex % 29792
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (59584 + x4 + (119168*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/js/cjsusgobk7jdjghdf42ayjqoamwrs55bw5376w7ziswcvxijbgbu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 304
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (119168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ka/cka2msdn4jdqyxuld5bv35vb2fc7giqhe2xblb3glgz45lpt27xe.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 59584)
    x4 = xindex % 59584
    x1 = (xindex // 196) % 304
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x4 + (119168*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cunt2ig5sn3talxgpjf563cjd3niqkmwmz6dliz6a3oz7j3sifn4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (225792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (225792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (225792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ru/crubjawpoh63yfjkreghaynlhnkusf277msrnte2hpudgv2dnn3r.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 288
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
    tmp9 = 0.00015943877551020407
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


# kernel path: /tmp/torchinductor_youkaichao/eo/ceo46sxz3l3xqybsgbj4ekidknz2mrowb2bvbvrn6y5syuepwnw4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (169344 + r1 + (784*x0) + (338688*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tmp12 * tmp14
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgysesml2z2kzry7f3zynhnpt3humonhek5fajyrruqusi7ljey.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 56448)
    x4 = xindex % 56448
    x1 = (xindex // 784) % 72
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (169344 + x4 + (338688*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00015943877551020407
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotlpaojipkpptsmq7m5xpvsq5jnrqu65exaiu3nvqshublhdsx5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqjdfkesqq5kncgvr6xnw4dwuwoenbc34a235bnmsefssbgcu3u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 144
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
    tmp9 = 0.00015943877551020407
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


# kernel path: /tmp/torchinductor_youkaichao/yp/cyp5hwzg2y2ms4ktmidmcqqvvazq7wyzntsp4ccrg4tv2qjexc3s.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (112896 + r1 + (784*x0) + (338688*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ep/cepvzxsvjdq6eoepipwxlyzhuy5shn63ywxo5clflfouhjvpcr52.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 56448)
    x4 = xindex % 56448
    x1 = (xindex // 784) % 72
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (112896 + x4 + (338688*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/cebopib2idbp5he3knpbfuv6lncxpf2aqy37eg6wsfgdaamermcd.py
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (338688*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3s/c3srz27zgohqmqpi6fohuxwgkzv53ztvilxudhts2w3c2xr4fmm4.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 112896)
    x4 = xindex % 112896
    x1 = (xindex // 784) % 144
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x4 + (338688*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqmcfnt37p5dtyt3lxq6t6yb2kgqxrtjdt5ygsql4lgxwu5l2ak.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (225792 + r1 + (784*x0) + (338688*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/66/c664zpcrzgthdjd7vd3aodq4nsllywocmmedy5kq4r2fzytl5ooa.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 112896)
    x4 = xindex % 112896
    x1 = (xindex // 784) % 144
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (225792 + x4 + (338688*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafdxx67f4jozasbyb33dgokgb226kjmpmrsocuolmta3sobsuog.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (169344 + r1 + (784*x0) + (225792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tmp12 * tmp14
    tl.store(out_ptr2 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w5/cw5xgqencudauoxz6ftdrrraffarjzf3dp4rr2u2gyhwgalw3fa5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 56448)
    x4 = xindex % 56448
    x1 = (xindex // 784) % 72
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (169344 + x4 + (225792*x2)), xmask)
    tmp4 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00015943877551020407
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwxdrfhxbrbhokjr26kwq3mbilak42s22zy3af4jogzs2qmwb3x.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (112896 + r1 + (784*x0) + (225792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/b4/cb4uzhah72ep2uyycne4sevwn5jbicyntgc4ki6la32dd7tph5qb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 56448)
    x4 = xindex % 56448
    x1 = (xindex // 784) % 72
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (112896 + x4 + (225792*x2)), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp7 = tl.load(in_ptr2 + (x3), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2rttkosvbvykm2eeikrpcazatztahuhkssqxo3to3td7pj6qtc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (225792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtkkyfrwxcr77yqrvoyagavlykj7ey5bvcja7lh62afd47rodmd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 112896)
    x4 = xindex % 112896
    x1 = (xindex // 784) % 144
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x4 + (225792*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/corziivtvssvhmsrqnsm5awav4bsgg5iqinudifxb6dqgwleqyd4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3wqflnr2jg67gzuui5iiaaj7sgk3ver7wbbq7puco5g4qcqdbv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/44/c44zx3an7nf2acu7jb7ieci3qyewjb2cnq5rmykq7yummqifdun7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pp/cppdn5hvubm2kkyszqgknuwx2omxh63ctqjex5th6eqt6bvhxd6f.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
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
    tmp9 = 3.985969387755102e-05
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


# kernel path: /tmp/torchinductor_youkaichao/qb/cqbzpam5fl7lzwzgywbvl76oiz44zyrvmwwp2wlsryohtobdxdbn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (301056 + (3136*x0) + (602112*(r2 // 3136)) + (1204224*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vpdv7eplrn4w2expu3vwb4j4kp7yzaoyf4tr5atlnrr4vlervr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/nk/cnk7krt5ctqz3jdhseiwtbefqcbisic4bpsue4cqjxyr6llm45h6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/caysom3fm7hcq44uom7i3fe5koicfucqhydoimmjrasvm5elqjkf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (301056 + x4 + (602112*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 3.985969387755102e-05
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cdsmcmmgd2yw552ocxfucijwy7qa3rrgv62ppf5f2e6nagyaqibp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/v5/cv5d6acqgscasyk55pwca2ja2sh3g4fnnz46jjdogjitf6faxhka.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qs/cqsgavy6xigfjprgtl4l336tcgku4zzrv23y5iqq5k2tssi54mjj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3vwlt7fzua2kmivtrlr3zd3xhct2nwiaandzd2ul3ittmukbxr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
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
    tmp9 = 3.985969387755102e-05
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


# kernel path: /tmp/torchinductor_youkaichao/hw/chwnipvvxbdyhcfaucy7ymtqxhq5nfboyssgy5pb66xxugacuscy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + (3136*x0) + (602112*(r2 // 3136)) + (1204224*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yn/cynurer5kyy4idoe6wtritskqoizgb2ywjrmaychkdgpkoiabxh5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (200704 + x4 + (602112*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbt4rqquzfltsc6mv3xlxuyuw2ixamg2ntdh65dv5up4mx226qc.py
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (602112*(r2 // 3136)) + (1204224*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqej3vdi6ou27zwtt7men6vtlifaf4b6kinwvwsepgyk4o6gwna.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 200704)
    x4 = xindex % 200704
    x1 = (xindex // 3136) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x4 + (602112*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvgoazjkmov7rn53swot27ekurildvodkunmefh5jxmcrru7n2td.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (401408 + (3136*x0) + (602112*(r2 // 3136)) + (1204224*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chy64muqrjxmb3xhcm277wdh7ykahlwe5g4s5mhwfjtd4qsqloin.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 200704)
    x4 = xindex % 200704
    x1 = (xindex // 3136) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (401408 + x4 + (602112*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ng/cngz6nf5euksmq4b6r4ul6gxnu5ivbbgqlgs7dg2a6yaqgidcau5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (301056 + (3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caok6o5icnkl3p5ihk2zt3hla6zm6oh4p6qnprlewvm5jceige6g.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (301056 + x4 + (401408*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 3.985969387755102e-05
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cneovnyqtg7x3dsoga4kprlx3eimkanimdi67m6czspqbji3ml5o.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + (3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggdo42s6wyd5t2d3eo4a4qqxrofzmu6h2t733yrqy4uzwdi2wye.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (200704 + x4 + (401408*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/corn6ft5zqn3ub24ghdovx4gerijxhnkomnlknuug2bn4gqsd4xn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5jbqwlbe3ygqasiqkw36o5wr32aydoyzxzwk5i6f6xiiitvjne.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 200704)
    x4 = xindex % 200704
    x1 = (xindex // 3136) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x4 + (401408*x2)), None)
    tmp4 = tl.load(in_out_ptr0 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpkjniajz2n3oxec4iaibuhdh2lcuodxjefsnkn64qszfjnouwnd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfw7rmzemzswcgykniyowrlif7nnx2hd5v56cpsbxprx67x3ejr7.py
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
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/cihccnucbwtk5fsppgjfvpwkvpiitu2jqawtne6eo56v435o52p4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4aa7xa7ztm7g6kk3cncn47bfzdn6f3f4jdbow7mdggtiqfypjq2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
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
    tmp9 = 9.964923469387754e-06
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
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_249, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, cat, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, relu_7, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, relu_10, convolution_11, squeeze_34, cat_1, convolution_12, squeeze_37, relu_12, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, relu_14, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, cat_2, convolution_18, squeeze_55, relu_18, convolution_19, squeeze_58, relu_19, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, relu_21, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, cat_3, convolution_24, squeeze_73, relu_24, convolution_25, squeeze_76, relu_25, convolution_26, squeeze_79, relu_26, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, cat_4, convolution_30, squeeze_91, relu_30, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, relu_33, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, cat_5, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, relu_38, convolution_39, squeeze_118, relu_39, convolution_40, squeeze_121, clone, permute_1, le, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, le_5, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, le_11, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, le_17, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, le_23, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, le_29, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, le_35, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_13, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_19, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_28, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_31, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_34, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_37, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_40, (144, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_41, (144, ), (1, ))
    assert_size_stride(primals_43, (144, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_44, (144, ), (1, ))
    assert_size_stride(primals_46, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_47, (72, ), (1, ))
    assert_size_stride(primals_49, (144, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_50, (144, ), (1, ))
    assert_size_stride(primals_52, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_53, (72, ), (1, ))
    assert_size_stride(primals_55, (144, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_56, (144, ), (1, ))
    assert_size_stride(primals_58, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_59, (144, ), (1, ))
    assert_size_stride(primals_61, (144, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_62, (144, ), (1, ))
    assert_size_stride(primals_64, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_65, (72, ), (1, ))
    assert_size_stride(primals_67, (144, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_68, (144, ), (1, ))
    assert_size_stride(primals_70, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_71, (72, ), (1, ))
    assert_size_stride(primals_73, (288, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(primals_74, (288, ), (1, ))
    assert_size_stride(primals_76, (304, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(primals_77, (304, ), (1, ))
    assert_size_stride(primals_79, (304, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_80, (304, ), (1, ))
    assert_size_stride(primals_82, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_83, (152, ), (1, ))
    assert_size_stride(primals_85, (304, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_86, (304, ), (1, ))
    assert_size_stride(primals_88, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_89, (152, ), (1, ))
    assert_size_stride(primals_91, (304, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_92, (304, ), (1, ))
    assert_size_stride(primals_94, (304, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_95, (304, ), (1, ))
    assert_size_stride(primals_97, (304, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_98, (304, ), (1, ))
    assert_size_stride(primals_100, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_101, (152, ), (1, ))
    assert_size_stride(primals_103, (304, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_104, (304, ), (1, ))
    assert_size_stride(primals_106, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_107, (152, ), (1, ))
    assert_size_stride(primals_109, (480, 912, 1, 1), (912, 1, 1, 1))
    assert_size_stride(primals_110, (480, ), (1, ))
    assert_size_stride(primals_112, (960, 480, 3, 3), (4320, 9, 3, 1))
    assert_size_stride(primals_113, (960, ), (1, ))
    assert_size_stride(primals_115, (1024, 960, 3, 3), (8640, 9, 3, 1))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_118, (1280, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_119, (1280, ), (1, ))
    assert_size_stride(primals_121, (1024, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_122, (1024, ), (1, ))
    assert_size_stride(primals_249, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(convolution_1, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_2, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_3, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_10, (32, ), (1, ))
    assert_size_stride(relu_3, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(convolution_4, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(relu_4, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_5, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_16, (32, ), (1, ))
    assert_size_stride(cat, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_6, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(relu_6, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_7, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(relu_7, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_8, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_8, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_9, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_28, (32, ), (1, ))
    assert_size_stride(relu_9, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(convolution_10, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(squeeze_31, (64, ), (1, ))
    assert_size_stride(relu_10, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_11, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_34, (32, ), (1, ))
    assert_size_stride(cat_1, (8, 192, 56, 56), (602112, 3136, 56, 1))
    assert_size_stride(convolution_12, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(relu_12, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_13, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(squeeze_40, (144, ), (1, ))
    assert_size_stride(relu_13, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(convolution_14, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(squeeze_43, (144, ), (1, ))
    assert_size_stride(relu_14, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(convolution_15, (8, 72, 28, 28), (56448, 784, 28, 1))
    assert_size_stride(squeeze_46, (72, ), (1, ))
    assert_size_stride(relu_15, (8, 72, 28, 28), (56448, 784, 28, 1))
    assert_size_stride(convolution_16, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(squeeze_49, (144, ), (1, ))
    assert_size_stride(relu_16, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(convolution_17, (8, 72, 28, 28), (56448, 784, 28, 1))
    assert_size_stride(squeeze_52, (72, ), (1, ))
    assert_size_stride(cat_2, (8, 288, 28, 28), (225792, 784, 28, 1))
    assert_size_stride(convolution_18, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(squeeze_55, (144, ), (1, ))
    assert_size_stride(relu_18, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(convolution_19, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(squeeze_58, (144, ), (1, ))
    assert_size_stride(relu_19, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(convolution_20, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(squeeze_61, (144, ), (1, ))
    assert_size_stride(relu_20, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(convolution_21, (8, 72, 28, 28), (56448, 784, 28, 1))
    assert_size_stride(squeeze_64, (72, ), (1, ))
    assert_size_stride(relu_21, (8, 72, 28, 28), (56448, 784, 28, 1))
    assert_size_stride(convolution_22, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(squeeze_67, (144, ), (1, ))
    assert_size_stride(relu_22, (8, 144, 28, 28), (112896, 784, 28, 1))
    assert_size_stride(convolution_23, (8, 72, 28, 28), (56448, 784, 28, 1))
    assert_size_stride(squeeze_70, (72, ), (1, ))
    assert_size_stride(cat_3, (8, 432, 28, 28), (338688, 784, 28, 1))
    assert_size_stride(convolution_24, (8, 288, 28, 28), (225792, 784, 28, 1))
    assert_size_stride(squeeze_73, (288, ), (1, ))
    assert_size_stride(relu_24, (8, 288, 28, 28), (225792, 784, 28, 1))
    assert_size_stride(convolution_25, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(squeeze_76, (304, ), (1, ))
    assert_size_stride(relu_25, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(convolution_26, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(squeeze_79, (304, ), (1, ))
    assert_size_stride(relu_26, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(convolution_27, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_82, (152, ), (1, ))
    assert_size_stride(relu_27, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_28, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(squeeze_85, (304, ), (1, ))
    assert_size_stride(relu_28, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(convolution_29, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_88, (152, ), (1, ))
    assert_size_stride(cat_4, (8, 608, 14, 14), (119168, 196, 14, 1))
    assert_size_stride(convolution_30, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(squeeze_91, (304, ), (1, ))
    assert_size_stride(relu_30, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(convolution_31, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(squeeze_94, (304, ), (1, ))
    assert_size_stride(relu_31, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(convolution_32, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(squeeze_97, (304, ), (1, ))
    assert_size_stride(relu_32, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(convolution_33, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_100, (152, ), (1, ))
    assert_size_stride(relu_33, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(convolution_34, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(squeeze_103, (304, ), (1, ))
    assert_size_stride(relu_34, (8, 304, 14, 14), (59584, 196, 14, 1))
    assert_size_stride(convolution_35, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(squeeze_106, (152, ), (1, ))
    assert_size_stride(cat_5, (8, 912, 14, 14), (178752, 196, 14, 1))
    assert_size_stride(convolution_36, (8, 480, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(squeeze_109, (480, ), (1, ))
    assert_size_stride(relu_36, (8, 480, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(convolution_37, (8, 960, 7, 7), (47040, 49, 7, 1))
    assert_size_stride(squeeze_112, (960, ), (1, ))
    assert_size_stride(relu_37, (8, 960, 7, 7), (47040, 49, 7, 1))
    assert_size_stride(convolution_38, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(squeeze_115, (1024, ), (1, ))
    assert_size_stride(relu_38, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_39, (8, 1280, 4, 4), (20480, 16, 4, 1))
    assert_size_stride(squeeze_118, (1280, ), (1, ))
    assert_size_stride(relu_39, (8, 1280, 4, 4), (20480, 16, 4, 1))
    assert_size_stride(convolution_40, (8, 1024, 4, 4), (16384, 16, 4, 1))
    assert_size_stride(squeeze_121, (1024, ), (1, ))
    assert_size_stride(clone, (8, 1024), (1024, 1))
    assert_size_stride(permute_1, (1000, 1024), (1024, 1))
    assert_size_stride(le, (8, 1024, 4, 4), (16384, 16, 4, 1))
    assert_size_stride(unsqueeze_166, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_190, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_202, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_214, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(le_5, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(unsqueeze_226, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(le_11, (8, 152, 14, 14), (29792, 196, 14, 1))
    assert_size_stride(unsqueeze_298, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(le_17, (8, 72, 28, 28), (56448, 784, 28, 1))
    assert_size_stride(unsqueeze_370, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(le_23, (8, 72, 28, 28), (56448, 784, 28, 1))
    assert_size_stride(unsqueeze_442, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_29, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_514, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_35, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_586, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf4 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf5 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_40, unsqueeze_166, squeeze_121, buf3, buf4, buf5, 1024, 128, grid=grid(1024), stream=stream0)
        buf6 = empty((8, 1024, 4, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2.run(le, buf0, convolution_40, unsqueeze_166, buf4, squeeze_121, buf3, primals_122, buf6, 131072, grid=grid(131072), stream=stream0)
        del buf0
        del convolution_40
        del le
        del primals_122
        del squeeze_121
        del unsqueeze_166
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf7 = aten.convolution_backward(buf6, relu_39, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf6
        del primals_121
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        buf10 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf11 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf12 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(relu_39, buf8, convolution_39, unsqueeze_178, squeeze_118, buf10, buf11, buf12, 1280, 128, grid=grid(1280), stream=stream0)
        buf13 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(buf13, relu_39, convolution_39, unsqueeze_178, buf11, squeeze_118, buf10, primals_119, 163840, grid=grid(163840), stream=stream0)
        del buf11
        del convolution_39
        del primals_119
        del relu_39
        del squeeze_118
        del unsqueeze_178
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf14 = aten.convolution_backward(buf13, relu_38, primals_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf13
        del primals_118
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = buf4; del buf4  # reuse
        buf18 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf19 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_5.run(relu_38, buf15, convolution_38, unsqueeze_190, squeeze_115, buf17, buf18, buf19, 1024, 392, grid=grid(1024), stream=stream0)
        buf20 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6.run(buf20, relu_38, convolution_38, unsqueeze_190, buf18, squeeze_115, buf17, primals_116, 401408, grid=grid(401408), stream=stream0)
        del buf18
        del convolution_38
        del primals_116
        del relu_38
        del squeeze_115
        del unsqueeze_190
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf21 = aten.convolution_backward(buf20, relu_37, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf20
        del primals_115
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = empty((960, ), device='cuda', dtype=torch.float32)
        buf25 = empty((960, ), device='cuda', dtype=torch.float32)
        buf26 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(relu_37, buf22, convolution_37, unsqueeze_202, squeeze_112, buf24, buf25, buf26, 960, 392, grid=grid(960), stream=stream0)
        buf27 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(buf27, relu_37, convolution_37, unsqueeze_202, buf25, squeeze_112, buf24, primals_113, 376320, grid=grid(376320), stream=stream0)
        del buf25
        del convolution_37
        del primals_113
        del relu_37
        del squeeze_112
        del unsqueeze_202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf28 = aten.convolution_backward(buf27, relu_36, primals_112, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf27
        del primals_112
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = empty((480, ), device='cuda', dtype=torch.float32)
        buf32 = empty((480, ), device='cuda', dtype=torch.float32)
        buf33 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_36, buf29, convolution_36, unsqueeze_214, squeeze_109, buf31, buf32, buf33, 480, 1568, grid=grid(480), stream=stream0)
        buf34 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(buf34, relu_36, convolution_36, unsqueeze_214, buf32, squeeze_109, buf31, primals_110, 752640, grid=grid(752640), stream=stream0)
        del buf32
        del convolution_36
        del primals_110
        del relu_36
        del squeeze_109
        del unsqueeze_214
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf35 = aten.convolution_backward(buf34, cat_5, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf34
        del cat_5
        del primals_109
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = empty((152, ), device='cuda', dtype=torch.float32)
        buf39 = empty((152, ), device='cuda', dtype=torch.float32)
        buf40 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_11.run(le_5, buf36, convolution_35, unsqueeze_226, squeeze_106, buf38, buf39, buf40, 152, 1568, grid=grid(152), stream=stream0)
        buf41 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(le_5, buf36, convolution_35, unsqueeze_226, buf39, squeeze_106, buf38, primals_107, buf41, 238336, grid=grid(238336), stream=stream0)
        del convolution_35
        del le_5
        del primals_107
        del squeeze_106
        del unsqueeze_226
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf42 = aten.convolution_backward(buf41, relu_34, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf41
        del primals_106
        buf43 = buf42[0]
        buf44 = buf42[1]
        del buf42
        buf45 = empty((304, ), device='cuda', dtype=torch.float32)
        buf46 = empty((304, ), device='cuda', dtype=torch.float32)
        buf47 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_13.run(relu_34, buf43, convolution_34, unsqueeze_238, squeeze_103, buf45, buf46, buf47, 304, 1568, grid=grid(304), stream=stream0)
        buf48 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf48, relu_34, convolution_34, unsqueeze_238, buf46, squeeze_103, buf45, primals_104, 476672, grid=grid(476672), stream=stream0)
        del convolution_34
        del primals_104
        del relu_34
        del squeeze_103
        del unsqueeze_238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf49 = aten.convolution_backward(buf48, relu_33, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf48
        del primals_103
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        buf52 = buf39; del buf39  # reuse
        buf53 = empty((152, ), device='cuda', dtype=torch.float32)
        buf55 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_15.run(relu_33, buf36, buf50, convolution_33, unsqueeze_250, squeeze_100, buf52, buf53, buf55, 152, 1568, grid=grid(152), stream=stream0)
        buf54 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_16.run(buf54, relu_33, buf36, convolution_33, unsqueeze_250, buf53, squeeze_100, buf52, primals_101, 238336, grid=grid(238336), stream=stream0)
        del convolution_33
        del primals_101
        del relu_33
        del squeeze_100
        del unsqueeze_250
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf56 = aten.convolution_backward(buf54, relu_32, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_100
        buf57 = buf56[0]
        buf58 = buf56[1]
        del buf56
        buf59 = buf46; del buf46  # reuse
        buf60 = empty((304, ), device='cuda', dtype=torch.float32)
        buf61 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_13.run(relu_32, buf57, convolution_32, unsqueeze_262, squeeze_97, buf59, buf60, buf61, 304, 1568, grid=grid(304), stream=stream0)
        buf62 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf62, relu_32, convolution_32, unsqueeze_262, buf60, squeeze_97, buf59, primals_98, 476672, grid=grid(476672), stream=stream0)
        del convolution_32
        del primals_98
        del relu_32
        del squeeze_97
        del unsqueeze_262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf63 = aten.convolution_backward(buf62, relu_31, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf62
        del primals_97
        buf64 = buf63[0]
        buf65 = buf63[1]
        del buf63
        buf66 = buf60; del buf60  # reuse
        buf67 = empty((304, ), device='cuda', dtype=torch.float32)
        buf69 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_17.run(relu_31, buf36, buf64, convolution_31, unsqueeze_274, squeeze_94, buf66, buf67, buf69, 304, 1568, grid=grid(304), stream=stream0)
        buf68 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_18.run(buf68, relu_31, buf36, convolution_31, unsqueeze_274, buf67, squeeze_94, buf66, primals_95, 476672, grid=grid(476672), stream=stream0)
        del convolution_31
        del primals_95
        del relu_31
        del squeeze_94
        del unsqueeze_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf70 = aten.convolution_backward(buf68, relu_30, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf68
        del primals_94
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = buf67; del buf67  # reuse
        buf74 = empty((304, ), device='cuda', dtype=torch.float32)
        buf76 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_19.run(relu_30, buf36, buf71, convolution_30, unsqueeze_286, squeeze_91, buf73, buf74, buf76, 304, 1568, grid=grid(304), stream=stream0)
        buf75 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_20.run(buf75, relu_30, buf36, convolution_30, unsqueeze_286, buf74, squeeze_91, buf73, primals_92, 476672, grid=grid(476672), stream=stream0)
        del buf36
        del convolution_30
        del primals_92
        del relu_30
        del squeeze_91
        del unsqueeze_286
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf77 = aten.convolution_backward(buf75, cat_4, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf75
        del cat_4
        del primals_91
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf80 = buf53; del buf53  # reuse
        buf81 = empty((152, ), device='cuda', dtype=torch.float32)
        buf82 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_21.run(le_11, buf78, convolution_29, unsqueeze_298, squeeze_88, buf80, buf81, buf82, 152, 1568, grid=grid(152), stream=stream0)
        buf83 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22.run(le_11, buf78, convolution_29, unsqueeze_298, buf81, squeeze_88, buf80, primals_89, buf83, 238336, grid=grid(238336), stream=stream0)
        del convolution_29
        del le_11
        del primals_89
        del squeeze_88
        del unsqueeze_298
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf84 = aten.convolution_backward(buf83, relu_28, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf83
        del primals_88
        buf85 = buf84[0]
        buf86 = buf84[1]
        del buf84
        buf87 = buf74; del buf74  # reuse
        buf88 = empty((304, ), device='cuda', dtype=torch.float32)
        buf89 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_13.run(relu_28, buf85, convolution_28, unsqueeze_310, squeeze_85, buf87, buf88, buf89, 304, 1568, grid=grid(304), stream=stream0)
        buf90 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf90, relu_28, convolution_28, unsqueeze_310, buf88, squeeze_85, buf87, primals_86, 476672, grid=grid(476672), stream=stream0)
        del convolution_28
        del primals_86
        del relu_28
        del squeeze_85
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf91 = aten.convolution_backward(buf90, relu_27, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf90
        del primals_85
        buf92 = buf91[0]
        buf93 = buf91[1]
        del buf91
        buf94 = buf81; del buf81  # reuse
        buf95 = empty((152, ), device='cuda', dtype=torch.float32)
        buf97 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_23.run(relu_27, buf78, buf92, convolution_27, unsqueeze_322, squeeze_82, buf94, buf95, buf97, 152, 1568, grid=grid(152), stream=stream0)
        buf96 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24.run(buf96, relu_27, buf78, convolution_27, unsqueeze_322, buf95, squeeze_82, buf94, primals_83, 238336, grid=grid(238336), stream=stream0)
        del buf95
        del convolution_27
        del primals_83
        del relu_27
        del squeeze_82
        del unsqueeze_322
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf98 = aten.convolution_backward(buf96, relu_26, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf96
        del primals_82
        buf99 = buf98[0]
        buf100 = buf98[1]
        del buf98
        buf101 = buf88; del buf88  # reuse
        buf102 = empty((304, ), device='cuda', dtype=torch.float32)
        buf103 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_13.run(relu_26, buf99, convolution_26, unsqueeze_334, squeeze_79, buf101, buf102, buf103, 304, 1568, grid=grid(304), stream=stream0)
        buf104 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf104, relu_26, convolution_26, unsqueeze_334, buf102, squeeze_79, buf101, primals_80, 476672, grid=grid(476672), stream=stream0)
        del convolution_26
        del primals_80
        del relu_26
        del squeeze_79
        del unsqueeze_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf105 = aten.convolution_backward(buf104, relu_25, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf104
        del primals_79
        buf106 = buf105[0]
        buf107 = buf105[1]
        del buf105
        buf108 = buf102; del buf102  # reuse
        buf109 = empty((304, ), device='cuda', dtype=torch.float32)
        buf111 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_25.run(relu_25, buf78, buf106, convolution_25, unsqueeze_346, squeeze_76, buf108, buf109, buf111, 304, 1568, grid=grid(304), stream=stream0)
        buf110 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_26.run(buf110, relu_25, buf78, convolution_25, unsqueeze_346, buf109, squeeze_76, buf108, primals_77, 476672, grid=grid(476672), stream=stream0)
        del buf109
        del buf78
        del convolution_25
        del primals_77
        del relu_25
        del squeeze_76
        del unsqueeze_346
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf112 = aten.convolution_backward(buf110, relu_24, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf110
        del primals_76
        buf113 = buf112[0]
        buf114 = buf112[1]
        del buf112
        buf115 = empty((288, ), device='cuda', dtype=torch.float32)
        buf116 = empty((288, ), device='cuda', dtype=torch.float32)
        buf117 = empty((288, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_24, buf113, convolution_24, unsqueeze_358, squeeze_73, buf115, buf116, buf117, 288, 6272, grid=grid(288), stream=stream0)
        buf118 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(buf118, relu_24, convolution_24, unsqueeze_358, buf116, squeeze_73, buf115, primals_74, 1806336, grid=grid(1806336), stream=stream0)
        del buf116
        del convolution_24
        del primals_74
        del relu_24
        del squeeze_73
        del unsqueeze_358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf119 = aten.convolution_backward(buf118, cat_3, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf118
        del cat_3
        del primals_73
        buf120 = buf119[0]
        buf121 = buf119[1]
        del buf119
        buf122 = empty((72, ), device='cuda', dtype=torch.float32)
        buf123 = empty((72, ), device='cuda', dtype=torch.float32)
        buf124 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_29.run(le_17, buf120, convolution_23, unsqueeze_370, squeeze_70, buf122, buf123, buf124, 72, 6272, grid=grid(72), stream=stream0)
        buf125 = empty((8, 72, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30.run(le_17, buf120, convolution_23, unsqueeze_370, buf123, squeeze_70, buf122, primals_71, buf125, 451584, grid=grid(451584), stream=stream0)
        del convolution_23
        del le_17
        del primals_71
        del squeeze_70
        del unsqueeze_370
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf126 = aten.convolution_backward(buf125, relu_22, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf125
        del primals_70
        buf127 = buf126[0]
        buf128 = buf126[1]
        del buf126
        buf129 = empty((144, ), device='cuda', dtype=torch.float32)
        buf130 = empty((144, ), device='cuda', dtype=torch.float32)
        buf131 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_31.run(relu_22, buf127, convolution_22, unsqueeze_382, squeeze_67, buf129, buf130, buf131, 144, 6272, grid=grid(144), stream=stream0)
        buf132 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32.run(buf132, relu_22, convolution_22, unsqueeze_382, buf130, squeeze_67, buf129, primals_68, 903168, grid=grid(903168), stream=stream0)
        del convolution_22
        del primals_68
        del relu_22
        del squeeze_67
        del unsqueeze_382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf133 = aten.convolution_backward(buf132, relu_21, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf132
        del primals_67
        buf134 = buf133[0]
        buf135 = buf133[1]
        del buf133
        buf136 = buf123; del buf123  # reuse
        buf137 = empty((72, ), device='cuda', dtype=torch.float32)
        buf139 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_33.run(relu_21, buf120, buf134, convolution_21, unsqueeze_394, squeeze_64, buf136, buf137, buf139, 72, 6272, grid=grid(72), stream=stream0)
        buf138 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_34.run(buf138, relu_21, buf120, convolution_21, unsqueeze_394, buf137, squeeze_64, buf136, primals_65, 451584, grid=grid(451584), stream=stream0)
        del convolution_21
        del primals_65
        del relu_21
        del squeeze_64
        del unsqueeze_394
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf140 = aten.convolution_backward(buf138, relu_20, primals_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_64
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        buf143 = buf130; del buf130  # reuse
        buf144 = empty((144, ), device='cuda', dtype=torch.float32)
        buf145 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_31.run(relu_20, buf141, convolution_20, unsqueeze_406, squeeze_61, buf143, buf144, buf145, 144, 6272, grid=grid(144), stream=stream0)
        buf146 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32.run(buf146, relu_20, convolution_20, unsqueeze_406, buf144, squeeze_61, buf143, primals_62, 903168, grid=grid(903168), stream=stream0)
        del convolution_20
        del primals_62
        del relu_20
        del squeeze_61
        del unsqueeze_406
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf147 = aten.convolution_backward(buf146, relu_19, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf146
        del primals_61
        buf148 = buf147[0]
        buf149 = buf147[1]
        del buf147
        buf150 = buf144; del buf144  # reuse
        buf151 = empty((144, ), device='cuda', dtype=torch.float32)
        buf153 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_19, buf120, buf148, convolution_19, unsqueeze_418, squeeze_58, buf150, buf151, buf153, 144, 6272, grid=grid(144), stream=stream0)
        buf152 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(buf152, relu_19, buf120, convolution_19, unsqueeze_418, buf151, squeeze_58, buf150, primals_59, 903168, grid=grid(903168), stream=stream0)
        del convolution_19
        del primals_59
        del relu_19
        del squeeze_58
        del unsqueeze_418
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf154 = aten.convolution_backward(buf152, relu_18, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf152
        del primals_58
        buf155 = buf154[0]
        buf156 = buf154[1]
        del buf154
        buf157 = buf151; del buf151  # reuse
        buf158 = empty((144, ), device='cuda', dtype=torch.float32)
        buf160 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_37.run(relu_18, buf120, buf155, convolution_18, unsqueeze_430, squeeze_55, buf157, buf158, buf160, 144, 6272, grid=grid(144), stream=stream0)
        buf159 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(buf159, relu_18, buf120, convolution_18, unsqueeze_430, buf158, squeeze_55, buf157, primals_56, 903168, grid=grid(903168), stream=stream0)
        del buf120
        del convolution_18
        del primals_56
        del relu_18
        del squeeze_55
        del unsqueeze_430
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf161 = aten.convolution_backward(buf159, cat_2, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf159
        del cat_2
        del primals_55
        buf162 = buf161[0]
        buf163 = buf161[1]
        del buf161
        buf164 = buf137; del buf137  # reuse
        buf165 = empty((72, ), device='cuda', dtype=torch.float32)
        buf166 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_23, buf162, convolution_17, unsqueeze_442, squeeze_52, buf164, buf165, buf166, 72, 6272, grid=grid(72), stream=stream0)
        buf167 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(le_23, buf162, convolution_17, unsqueeze_442, buf165, squeeze_52, buf164, primals_53, buf167, 451584, grid=grid(451584), stream=stream0)
        del convolution_17
        del le_23
        del primals_53
        del squeeze_52
        del unsqueeze_442
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf168 = aten.convolution_backward(buf167, relu_16, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf167
        del primals_52
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = buf158; del buf158  # reuse
        buf172 = empty((144, ), device='cuda', dtype=torch.float32)
        buf173 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_31.run(relu_16, buf169, convolution_16, unsqueeze_454, squeeze_49, buf171, buf172, buf173, 144, 6272, grid=grid(144), stream=stream0)
        buf174 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32.run(buf174, relu_16, convolution_16, unsqueeze_454, buf172, squeeze_49, buf171, primals_50, 903168, grid=grid(903168), stream=stream0)
        del convolution_16
        del primals_50
        del relu_16
        del squeeze_49
        del unsqueeze_454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf175 = aten.convolution_backward(buf174, relu_15, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf174
        del primals_49
        buf176 = buf175[0]
        buf177 = buf175[1]
        del buf175
        buf178 = buf165; del buf165  # reuse
        buf179 = empty((72, ), device='cuda', dtype=torch.float32)
        buf181 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_41.run(relu_15, buf162, buf176, convolution_15, unsqueeze_466, squeeze_46, buf178, buf179, buf181, 72, 6272, grid=grid(72), stream=stream0)
        buf180 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42.run(buf180, relu_15, buf162, convolution_15, unsqueeze_466, buf179, squeeze_46, buf178, primals_47, 451584, grid=grid(451584), stream=stream0)
        del buf179
        del convolution_15
        del primals_47
        del relu_15
        del squeeze_46
        del unsqueeze_466
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf182 = aten.convolution_backward(buf180, relu_14, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf180
        del primals_46
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        buf185 = buf172; del buf172  # reuse
        buf186 = empty((144, ), device='cuda', dtype=torch.float32)
        buf187 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_31.run(relu_14, buf183, convolution_14, unsqueeze_478, squeeze_43, buf185, buf186, buf187, 144, 6272, grid=grid(144), stream=stream0)
        buf188 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32.run(buf188, relu_14, convolution_14, unsqueeze_478, buf186, squeeze_43, buf185, primals_44, 903168, grid=grid(903168), stream=stream0)
        del convolution_14
        del primals_44
        del relu_14
        del squeeze_43
        del unsqueeze_478
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf189 = aten.convolution_backward(buf188, relu_13, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf188
        del primals_43
        buf190 = buf189[0]
        buf191 = buf189[1]
        del buf189
        buf192 = buf186; del buf186  # reuse
        buf193 = empty((144, ), device='cuda', dtype=torch.float32)
        buf195 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_43.run(relu_13, buf162, buf190, convolution_13, unsqueeze_490, squeeze_40, buf192, buf193, buf195, 144, 6272, grid=grid(144), stream=stream0)
        buf194 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_44.run(buf194, relu_13, buf162, convolution_13, unsqueeze_490, buf193, squeeze_40, buf192, primals_41, 903168, grid=grid(903168), stream=stream0)
        del buf162
        del buf193
        del convolution_13
        del primals_41
        del relu_13
        del squeeze_40
        del unsqueeze_490
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf196 = aten.convolution_backward(buf194, relu_12, primals_40, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf194
        del primals_40
        buf197 = buf196[0]
        buf198 = buf196[1]
        del buf196
        buf199 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf201 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_12, buf197, convolution_12, unsqueeze_502, buf199, buf201, 512, 6272, grid=grid(512), stream=stream0)
        buf200 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf199, buf200, 128, 4, grid=grid(128), stream=stream0)
        del buf199
        buf202 = empty((128, ), device='cuda', dtype=torch.float32)
        buf203 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf201, squeeze_37, buf202, buf203, 128, 4, grid=grid(128), stream=stream0)
        del buf201
        buf204 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48.run(buf204, relu_12, convolution_12, unsqueeze_502, buf202, squeeze_37, buf200, primals_38, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_12
        del primals_38
        del relu_12
        del squeeze_37
        del unsqueeze_502
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf205 = aten.convolution_backward(buf204, cat_1, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf204
        del cat_1
        del primals_37
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = reinterpret_tensor(buf202, (32, 4), (1, 32), 0); del buf202  # reuse
        buf210 = empty_strided((32, 4), (1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(le_29, buf206, convolution_11, unsqueeze_514, buf208, buf210, 128, 6272, grid=grid(128), stream=stream0)
        buf209 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_50.run(buf208, buf209, 32, 4, grid=grid(32), stream=stream0)
        buf211 = empty((32, ), device='cuda', dtype=torch.float32)
        buf212 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_51.run(buf210, squeeze_34, buf211, buf212, 32, 4, grid=grid(32), stream=stream0)
        buf213 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52.run(le_29, buf206, convolution_11, unsqueeze_514, buf211, squeeze_34, buf209, primals_35, buf213, 802816, grid=grid(802816), stream=stream0)
        del convolution_11
        del le_29
        del primals_35
        del squeeze_34
        del unsqueeze_514
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf214 = aten.convolution_backward(buf213, relu_10, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf213
        del primals_34
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        buf217 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        buf219 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(relu_10, buf215, convolution_10, unsqueeze_526, buf217, buf219, 256, 6272, grid=grid(256), stream=stream0)
        buf218 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf217, buf218, 64, 4, grid=grid(64), stream=stream0)
        buf220 = empty((64, ), device='cuda', dtype=torch.float32)
        buf221 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_55.run(buf219, squeeze_31, buf220, buf221, 64, 4, grid=grid(64), stream=stream0)
        buf222 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf222, relu_10, convolution_10, unsqueeze_526, buf220, squeeze_31, buf218, primals_32, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_10
        del primals_32
        del relu_10
        del squeeze_31
        del unsqueeze_526
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf223 = aten.convolution_backward(buf222, relu_9, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf222
        del primals_31
        buf224 = buf223[0]
        buf225 = buf223[1]
        del buf223
        buf226 = buf210; del buf210  # reuse
        buf228 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_57.run(relu_9, buf206, buf224, convolution_9, unsqueeze_538, buf226, buf228, 128, 6272, grid=grid(128), stream=stream0)
        buf227 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_50.run(buf226, buf227, 32, 4, grid=grid(32), stream=stream0)
        buf229 = empty((32, ), device='cuda', dtype=torch.float32)
        buf231 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_51.run(buf228, squeeze_28, buf229, buf231, 32, 4, grid=grid(32), stream=stream0)
        buf230 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_58.run(buf230, relu_9, buf206, convolution_9, unsqueeze_538, buf229, squeeze_28, buf227, primals_29, 802816, grid=grid(802816), stream=stream0)
        del convolution_9
        del primals_29
        del relu_9
        del squeeze_28
        del unsqueeze_538
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf232 = aten.convolution_backward(buf230, relu_8, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_28
        buf233 = buf232[0]
        buf234 = buf232[1]
        del buf232
        buf235 = buf219; del buf219  # reuse
        buf237 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(relu_8, buf233, convolution_8, unsqueeze_550, buf235, buf237, 256, 6272, grid=grid(256), stream=stream0)
        buf236 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf235, buf236, 64, 4, grid=grid(64), stream=stream0)
        buf238 = empty((64, ), device='cuda', dtype=torch.float32)
        buf239 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_55.run(buf237, squeeze_25, buf238, buf239, 64, 4, grid=grid(64), stream=stream0)
        buf240 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf240, relu_8, convolution_8, unsqueeze_550, buf238, squeeze_25, buf236, primals_26, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_8
        del primals_26
        del relu_8
        del squeeze_25
        del unsqueeze_550
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf241 = aten.convolution_backward(buf240, relu_7, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf240
        del primals_25
        buf242 = buf241[0]
        buf243 = buf241[1]
        del buf241
        buf244 = buf237; del buf237  # reuse
        buf246 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_59.run(relu_7, buf206, buf242, convolution_7, unsqueeze_562, buf244, buf246, 256, 6272, grid=grid(256), stream=stream0)
        buf245 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf244, buf245, 64, 4, grid=grid(64), stream=stream0)
        buf247 = empty((64, ), device='cuda', dtype=torch.float32)
        buf249 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_55.run(buf246, squeeze_22, buf247, buf249, 64, 4, grid=grid(64), stream=stream0)
        buf248 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_60.run(buf248, relu_7, buf206, convolution_7, unsqueeze_562, buf247, squeeze_22, buf245, primals_23, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_7
        del primals_23
        del relu_7
        del squeeze_22
        del unsqueeze_562
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf250 = aten.convolution_backward(buf248, relu_6, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf248
        del primals_22
        buf251 = buf250[0]
        buf252 = buf250[1]
        del buf250
        buf253 = buf246; del buf246  # reuse
        buf255 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_61.run(relu_6, buf206, buf251, convolution_6, unsqueeze_574, buf253, buf255, 256, 6272, grid=grid(256), stream=stream0)
        buf254 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf253, buf254, 64, 4, grid=grid(64), stream=stream0)
        buf256 = empty((64, ), device='cuda', dtype=torch.float32)
        buf258 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_55.run(buf255, squeeze_19, buf256, buf258, 64, 4, grid=grid(64), stream=stream0)
        buf257 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_62.run(buf257, relu_6, buf206, convolution_6, unsqueeze_574, buf256, squeeze_19, buf254, primals_20, 1605632, grid=grid(1605632), stream=stream0)
        del buf206
        del convolution_6
        del primals_20
        del relu_6
        del squeeze_19
        del unsqueeze_574
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf259 = aten.convolution_backward(buf257, cat, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf257
        del cat
        del primals_19
        buf260 = buf259[0]
        buf261 = buf259[1]
        del buf259
        buf262 = buf228; del buf228  # reuse
        buf264 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_63.run(le_35, buf260, convolution_5, unsqueeze_586, buf262, buf264, 128, 6272, grid=grid(128), stream=stream0)
        buf263 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_50.run(buf262, buf263, 32, 4, grid=grid(32), stream=stream0)
        buf265 = empty((32, ), device='cuda', dtype=torch.float32)
        buf266 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_51.run(buf264, squeeze_16, buf265, buf266, 32, 4, grid=grid(32), stream=stream0)
        buf267 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64.run(le_35, buf260, convolution_5, unsqueeze_586, buf265, squeeze_16, buf263, primals_17, buf267, 802816, grid=grid(802816), stream=stream0)
        del convolution_5
        del le_35
        del primals_17
        del squeeze_16
        del unsqueeze_586
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf268 = aten.convolution_backward(buf267, relu_4, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf267
        del primals_16
        buf269 = buf268[0]
        buf270 = buf268[1]
        del buf268
        buf271 = buf255; del buf255  # reuse
        buf273 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(relu_4, buf269, convolution_4, unsqueeze_598, buf271, buf273, 256, 6272, grid=grid(256), stream=stream0)
        buf272 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf271, buf272, 64, 4, grid=grid(64), stream=stream0)
        buf274 = empty((64, ), device='cuda', dtype=torch.float32)
        buf275 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_55.run(buf273, squeeze_13, buf274, buf275, 64, 4, grid=grid(64), stream=stream0)
        buf276 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf276, relu_4, convolution_4, unsqueeze_598, buf274, squeeze_13, buf272, primals_14, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_4
        del primals_14
        del relu_4
        del squeeze_13
        del unsqueeze_598
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf277 = aten.convolution_backward(buf276, relu_3, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf276
        del primals_13
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = buf264; del buf264  # reuse
        buf282 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_65.run(relu_3, buf260, buf278, convolution_3, unsqueeze_610, buf280, buf282, 128, 6272, grid=grid(128), stream=stream0)
        buf281 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_50.run(buf280, buf281, 32, 4, grid=grid(32), stream=stream0)
        del buf280
        buf283 = empty((32, ), device='cuda', dtype=torch.float32)
        buf285 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_51.run(buf282, squeeze_10, buf283, buf285, 32, 4, grid=grid(32), stream=stream0)
        del buf282
        buf284 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_66.run(buf284, relu_3, buf260, convolution_3, unsqueeze_610, buf283, squeeze_10, buf281, primals_11, 802816, grid=grid(802816), stream=stream0)
        del convolution_3
        del primals_11
        del relu_3
        del squeeze_10
        del unsqueeze_610
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf286 = aten.convolution_backward(buf284, relu_2, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf284
        del primals_10
        buf287 = buf286[0]
        buf288 = buf286[1]
        del buf286
        buf289 = buf273; del buf273  # reuse
        buf291 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(relu_2, buf287, convolution_2, unsqueeze_622, buf289, buf291, 256, 6272, grid=grid(256), stream=stream0)
        buf290 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf289, buf290, 64, 4, grid=grid(64), stream=stream0)
        buf292 = empty((64, ), device='cuda', dtype=torch.float32)
        buf293 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_55.run(buf291, squeeze_7, buf292, buf293, 64, 4, grid=grid(64), stream=stream0)
        buf294 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf294, relu_2, convolution_2, unsqueeze_622, buf292, squeeze_7, buf290, primals_8, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_2
        del primals_8
        del relu_2
        del squeeze_7
        del unsqueeze_622
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf295 = aten.convolution_backward(buf294, relu_1, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf294
        del primals_7
        buf296 = buf295[0]
        buf297 = buf295[1]
        del buf295
        buf298 = buf291; del buf291  # reuse
        buf300 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_67.run(relu_1, buf260, buf296, convolution_1, unsqueeze_634, buf298, buf300, 256, 6272, grid=grid(256), stream=stream0)
        buf299 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf298, buf299, 64, 4, grid=grid(64), stream=stream0)
        del buf298
        buf301 = empty((64, ), device='cuda', dtype=torch.float32)
        buf303 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_55.run(buf300, squeeze_4, buf301, buf303, 64, 4, grid=grid(64), stream=stream0)
        del buf300
        buf302 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_68.run(buf302, relu_1, buf260, convolution_1, unsqueeze_634, buf301, squeeze_4, buf299, primals_5, 1605632, grid=grid(1605632), stream=stream0)
        del buf260
        del buf301
        del convolution_1
        del primals_5
        del relu_1
        del squeeze_4
        del unsqueeze_634
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf304 = aten.convolution_backward(buf302, relu, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf302
        del primals_4
        buf305 = buf304[0]
        buf306 = buf304[1]
        del buf304
        buf307 = empty((32, 13), device='cuda', dtype=torch.float32)
        buf309 = empty((32, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(relu, buf305, convolution, unsqueeze_646, buf307, buf309, 416, 7720, grid=grid(416), stream=stream0)
        buf308 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_70.run(buf307, buf308, 32, 13, grid=grid(32), stream=stream0)
        del buf307
        buf310 = empty((32, ), device='cuda', dtype=torch.float32)
        buf311 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_71.run(buf309, squeeze_1, buf310, buf311, 32, 13, grid=grid(32), stream=stream0)
        del buf309
        buf312 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72.run(buf312, relu, convolution, unsqueeze_646, buf310, squeeze_1, buf308, primals_2, 3211264, grid=grid(3211264), stream=stream0)
        del buf310
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_646
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf313 = aten.convolution_backward(buf312, primals_249, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf312
        del primals_1
        del primals_249
        buf314 = buf313[1]
        return (buf314, buf311, buf308, buf306, buf303, buf299, buf297, buf293, buf290, buf288, buf285, buf281, buf279, buf275, buf272, buf270, buf266, buf263, buf261, buf258, buf254, buf252, buf249, buf245, buf243, buf239, buf236, buf234, buf231, buf227, buf225, buf221, buf218, buf216, buf212, buf209, buf207, buf203, buf200, buf198, buf195, buf192, buf191, buf187, buf185, buf184, buf181, buf178, buf177, buf173, buf171, buf170, buf166, buf164, buf163, buf160, buf157, buf156, buf153, buf150, buf149, buf145, buf143, buf142, buf139, buf136, buf135, buf131, buf129, buf128, buf124, buf122, buf121, buf117, buf115, buf114, buf111, buf108, buf107, buf103, buf101, buf100, buf97, buf94, buf93, buf89, buf87, buf86, buf82, buf80, buf79, buf76, buf73, buf72, buf69, buf66, buf65, buf61, buf59, buf58, buf55, buf52, buf51, buf47, buf45, buf44, buf40, buf38, buf37, buf33, buf31, buf30, buf26, buf24, buf23, buf19, buf17, buf16, buf12, buf10, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((144, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((144, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((144, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((144, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((144, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((144, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((288, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((304, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((304, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((304, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((304, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((304, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((304, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((304, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((480, 912, 1, 1), (912, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((960, 480, 3, 3), (4320, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1024, 960, 3, 3), (8640, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1280, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 192, 56, 56), (602112, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 72, 28, 28), (56448, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 72, 28, 28), (56448, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 72, 28, 28), (56448, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 288, 28, 28), (225792, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 72, 28, 28), (56448, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((8, 72, 28, 28), (56448, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 144, 28, 28), (112896, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 72, 28, 28), (56448, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 432, 28, 28), (338688, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 288, 28, 28), (225792, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((8, 288, 28, 28), (225792, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 608, 14, 14), (119168, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 304, 14, 14), (59584, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 912, 14, 14), (178752, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 480, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((8, 480, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 960, 7, 7), (47040, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 960, 7, 7), (47040, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1280, 4, 4), (20480, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((8, 1280, 4, 4), (20480, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 1024, 4, 4), (16384, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 1024, 4, 4), (16384, 16, 4, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_166 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_5 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_226 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((8, 152, 14, 14), (29792, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_298 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((8, 72, 28, 28), (56448, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_370 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_23 = rand_strided((8, 72, 28, 28), (56448, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_442 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_29 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_514 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_35 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_586 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_249, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, cat, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, relu_7, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, relu_10, convolution_11, squeeze_34, cat_1, convolution_12, squeeze_37, relu_12, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, relu_14, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, cat_2, convolution_18, squeeze_55, relu_18, convolution_19, squeeze_58, relu_19, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, relu_21, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, cat_3, convolution_24, squeeze_73, relu_24, convolution_25, squeeze_76, relu_25, convolution_26, squeeze_79, relu_26, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, cat_4, convolution_30, squeeze_91, relu_30, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, relu_33, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, cat_5, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, relu_38, convolution_39, squeeze_118, relu_39, convolution_40, squeeze_121, clone, permute_1, le, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, le_5, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, le_11, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, le_17, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, le_23, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, le_29, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, le_35, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('selecsls42b', benchmark_compiled_module)
