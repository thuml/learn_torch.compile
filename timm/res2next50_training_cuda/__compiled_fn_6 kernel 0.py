
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


# kernel path: /tmp/torchinductor_youkaichao/5c/c5ck7uyonbkvn3c7zpbiyu6fi6hvdpmvibmjal3fin7q3ff6bufe.py
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = 49.0
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


# kernel path: /tmp/torchinductor_youkaichao/ez/cezou42lsx4xoe26dd67mtss6lt5n3l5gua536rwqubucgkg4yiz.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlqajlspwimrz7xaxei5b3nccownvgmeatrewtcejaf2d55ckgq.py
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
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (25088 + r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr2 + (x0), tmp17, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqb3aht24kwjpasjnuegcvfqarwg3nuwkn2haxj4gr5t6wbcrirx.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (25088 + x4 + (50176*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/7x/c7x2syq7dyfunmz3eaomwts3rpbmt2snibgtff675x6uxbjvbl4a.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (12544 + r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jg/cjgmtbk6xp3hvzgonjpyc2yvkixwdn6gkus4pz5jfkpzgu47n3zq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (12544 + x4 + (50176*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/m5/cm527giunc62gqliuinbpluklng2lfdz5wx6uwykfm5yviwrsqfo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp12 = tmp10 - tmp11
    tmp13 = tmp5 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rrhly5be25k4ua3uzerg7hdf44pqj67cf5nl7pfgugffvdxwrv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4 + (50176*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2dtm455prz4l6mooneg53yae4dmguapjojktbltnsajf3uscno.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (12544*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-12544) + x4 + (12544*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 768, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-25088) + x4 + (12544*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 1024, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_out_ptr0 + (x3), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbyp4qcpodw5hbm6mftqsmezc4jqu6dkixilbue6whpwsbf2afpm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7aayg6g3obzlpniikacrw36vhdlliit7f6n3rdufjg4y3aoyjd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/du/cdu7sltfjauh2htynvxppilaf6efw5n47x7cnhovcivft4m3lc7l.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (100352*r2)), rmask).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 49.0
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


# kernel path: /tmp/torchinductor_youkaichao/mw/cmwekdnp7ylvqndwkvxcxt6hrasf373verj4s6ldm3tmdr4d2foh.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 2048
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
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/az/cazxkoedxgtbuh3noejvjboejnxzzc3fqydwubqgbof52cuo5php.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]

triton_poi_fused_add_div_threshold_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None).to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr0 + (x2), None)
    tmp13 = tl.load(in_ptr4 + (x2), None)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3dp643ttsjluwaqg4ma3d53rs55qwqmgaovxcj6qnrxn27t2zh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5luiacmnmlc7af3bnsxqubguxckac6kouykobhamf5ys56wtxtn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
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
    tmp5 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/hf/chfjod5a7ehbbd37cyfvdd6z3holblzns7zkp4a35m5jp5fstr65.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196) % 256
    x3 = (xindex // 50176)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (37632 + (7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (50176*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (37632 + (7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (50176*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), None)
    tmp18 = tl.load(in_ptr0 + (37632 + (7*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (50176*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (37632 + (7*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (50176*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(7, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(7, 1 + ((1 + x0) // 2))
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
    tl.store(out_ptr0 + (x6), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vo/cvow76ayvb4sf3bubod7jb44rxup25xh4n6kak3utwejs6jgfosp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (12544 + r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr2 + (x0), tmp17, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xjkhkpkb2tqkbaaf7fvyojskhjriqwhzvwjxnc3567awn7q3pq.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (12544 + x4 + (50176*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/33/c33m5tx6hpp2p4rqp33zahgiewoibcek4r4dadpply4dvrzruwk7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr2 + (x0), tmp17, xmask)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmub5jdwj6jrxawfzag53jzqfi6zdtxpokktxomqdjlmu3xckih.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    x1 = (xindex // 49) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4 + (50176*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/6e/c6ehcmj7pnyeqofshs72pt3abqb5if4vlfjv7ojmeiklsudxbwu4.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    x2 = (xindex // 200704)
    x4 = xindex % 200704
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (50176*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-50176) + x4 + (50176*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 768, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-100352) + x4 + (50176*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 1024, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-150528) + x4 + (50176*x2)), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/cow64l3y3pvt43oshfk6saiqamph3ywisqwhwzgb3jdtalikmhzi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2qmrml7uvhyigrl5vs5b4puidsrntq3e5jk3djrrubzgc7ptfu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
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


# kernel path: /tmp/torchinductor_youkaichao/on/consgelkmyw6723z7auib46hvqkvymnc3cjgdxgynpxlf3dhy7yv.py
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfd5veradn47nisxmiigh3mxstc3obgtsja3vht7w6bszobxzdg.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
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
    tl.store(out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eo/ceoelzdx6cln5qjv2huijx4ia4s76p73qso2jw3it2d56nf3uau6.py
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
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (50176 + r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kapplcpjxt2nqgef3s6zurpobnokl53ojob6gukxazjdkrw4vv.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (50176 + x4 + (100352*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dqvqbdpph37w45cxu4a3s7zwp43hhxwuwbgpzwj67gtzp2uujd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (25088 + r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr2 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymam2zyymmb36jioikgl2sdn4r2eklduvxqkvc6bfez2p2f2ivs.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (25088 + x4 + (100352*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.0006377551020408163
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


# kernel path: /tmp/torchinductor_youkaichao/v3/cv3pfozil7xfc2yh522mawemoftybtsdi3vtjl4ofmgs2nfsgvjr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr2 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uk/cuk47ee7b2v3q6fvuhawwjiqhccvn7przw2zg3j73h7y5mdl3f3v.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4 + (100352*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.0006377551020408163
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


# kernel path: /tmp/torchinductor_youkaichao/am/camjnrlxxzfxk4az5xyhe3vl7hujtsxyzzw342mkg2aerwtwd6f5.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 128, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (25088*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 256, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-25088) + x4 + (25088*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 384, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-50176) + x4 + (25088*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 512, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_out_ptr0 + (x3), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/cams7fgrpvoec4pklspcuity2fjdnjqo2fwl3ckwf7jbllzlcp77.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bg/cbgguc3n4zflcbc347sgdjddbblhhc23kqetmyqqvwex3a4v5otn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
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


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4szl2thyg3vydg6bnignvui7ktmazwlbwbrljyu37tzdam754o.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp6 = tl.load(in_out_ptr0 + (x0), None)
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


# kernel path: /tmp/torchinductor_youkaichao/7p/c7pr4unqw5hlwtr6eylc4fxmtkyoczq6yw7azmmv75ohqx64yszw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
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


# kernel path: /tmp/torchinductor_youkaichao/x4/cx45xlgvch6sboau45kluhxeipkaalxizzlusz6hhmgmeps4lgtb.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_youkaichao/6h/c6h2xdwexup45lku2cw42ncw267ybkmy4irwlgjdtt3jgtx5vbwe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ke/ckeeqon7l5nzhrpcnfpgxxsg5p42dactayxuxy36gddprmpe2wak.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
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
    tmp5 = 0.0006377551020408163
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


# kernel path: /tmp/torchinductor_youkaichao/5d/c5d5oqomehv2eohxyjw7mypmwosfubprwipmzeuqqe2qxew77kbn.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784) % 128
    x3 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (75264 + (14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (100352*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (75264 + (14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (100352*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), None)
    tmp18 = tl.load(in_ptr0 + (75264 + (14*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (100352*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (75264 + (14*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (100352*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(14, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(14, 1 + ((1 + x0) // 2))
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
    tl.store(out_ptr0 + (x6), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/as/casvorz5tt5nmffffpzhbyqcmwmo6ewj7zo66dfu7ixld7ymiqy4.py
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
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (25088 + r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hy/chyzp24pd6ak6yjm5yy5ur7wguhwspthvv4tj4zzfvpndvkq6gji.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (25088 + x4 + (100352*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clkm77miqds2qlp2swfhzwxhpzsunnfp6y6hpro4dh5qdppkaryx.py
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
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4o/c4ok2jxmb4ecpaia4iilzzgmvutx3745eja4jbsmh4uo4gdlgo4g.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4 + (100352*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4al24waxdf3aql7detirqqa2npudvda7775wzemq5ck5y7ovx6k.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 128, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (100352*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 256, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-100352) + x4 + (100352*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 384, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-200704) + x4 + (100352*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 512, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-301056) + x4 + (100352*x2)), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zao26qanagbnf6ako45l2yflf7ut7mi2y7ymzmzenjbllrjstn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5wq3fbhgb4ug5tlapddfaxsgyvgoqkgqqswu3n2turr5xprt3d.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
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


# kernel path: /tmp/torchinductor_youkaichao/o5/co5r2fwgdbg7qurvpxapkwhr7xiacxb5sohuwwknmf74rbdhsfpt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gvbkxzjmzlnavqyyfy5v35o5r7oj7rfa6tcewhy5qo2xndtvo4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
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
    tl.store(out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedbn2gafomfksqmz5iqxjdqfhijtstu5zdorggyokag3bio2u4k.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (100352 + r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjnlh4uea5z2eocbvect6jl5fllkkzfkiygx7x7p7t3xi3ujlot.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (100352 + x4 + (200704*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjn7rlv6a2g24cwp3kkxwb6up42iravgssjb5cpxoxyswnskuix5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (50176 + r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr2 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn34fah5lomb3fgacmcto2xa2pdrph6hm643m4x6fxetklsp4rb4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (50176 + x4 + (200704*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
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


# kernel path: /tmp/torchinductor_youkaichao/2c/c2c5xtidqad47jzxlqoqikjwxyykzsxmsnaxr3i5comrmpzvb3ov.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr2 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ca/cca3yr62gz322erpjq54wgdqk4pdgxv24oivobeyauv6atvuwxpe.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4 + (200704*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
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


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7zag3tb74prdbmxcdqoact37ssxavbmax2qtmxulrwotnt62yy.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    x2 = (xindex // 200704)
    x4 = xindex % 200704
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 64, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (50176*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 128, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-50176) + x4 + (50176*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 192, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-100352) + x4 + (50176*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 256, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_out_ptr0 + (x3), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjo7cvftlx4e3a7ooisjhyafk4yo3knpkns7xqjg25jc7djjrmlb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6y/c6y7dp4ho6facv2kmjd4ujmzbuu4sfg5rllz4ydjx62fljqxa57l.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
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


# kernel path: /tmp/torchinductor_youkaichao/5s/c5st3lay3ptyytdgiuz6zaw7rc32vy6sevlv6hwirael2anwu5zb.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmudydf5z24suypyap5vvzu3iu2wb34fxatt6gbbwr2a52idyah.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
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


# kernel path: /tmp/torchinductor_youkaichao/ot/cotq4qwua7fdrkc2fj6zooh3sz2tctq6iznlqv6c2w7ejy64azxs.py
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
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jw/cjwp25b42jgtcjf4vacvopjmwcdfrjiqzpyvuio34d3latszb4u2.py
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
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
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
    tmp5 = 0.00015943877551020407
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


# kernel path: /tmp/torchinductor_youkaichao/k7/ck74364a56x3suvamjf76nyvzqkdvetdyjdjzzbjl5oe4dw3vlyy.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136) % 64
    x3 = (xindex // 200704)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (150528 + (28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (200704*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (150528 + (28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (200704*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None)
    tmp18 = tl.load(in_ptr0 + (150528 + (28*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (200704*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (150528 + (28*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (200704*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(28, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(28, 1 + ((1 + x0) // 2))
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
    tl.store(out_ptr0 + (x6), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camj26vsgljkgi7jeds2jsuicat3rfjh64pafyjppp23oj54bfed.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (50176 + r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/id/cidjlz5k4hbtaxhi2lf7skndq7tgayed7u4w3qerw5y62dqoka7q.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (50176 + x4 + (200704*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjc3n7yswxgzbzacbgvc2mnynfbt254k2ayged5fbkz4d6outns.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jz/cjzhfkwteafups33ie5q5d2jvsd4smo4wu7cbiulnvbsttqh67mj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4 + (200704*x2)), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqapeggxehu674x4sm2emm22wjz4dncw7ld7z3wffleh6l3sbjpz.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    x2 = (xindex // 802816)
    x4 = xindex % 802816
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 64, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (200704*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 128, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-200704) + x4 + (200704*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 192, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-401408) + x4 + (200704*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 256, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-602112) + x4 + (200704*x2)), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnxpud7qxshguh65t4f775qi533554cfvc4viwrcoilqzxmw4f5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqghdx4y4qy3ztykyndffl7ouaz6rpd3emwagvw6zsl6clgfaso.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 3.985969387755102e-05
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


# kernel path: /tmp/torchinductor_youkaichao/dz/cdzwld6dlwsjj74c4t674qsiejujbtf447t2hatd7i4j4syanfoe.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/r4/cr4qtif3b2ikpw75iwjbe33xqae5qac6xmvwgbntcb3ny5llpy5b.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
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
    tl.store(out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4fgrs4ads765wmiiwmpiibzdijk6fg6yiyvwhxeved6d3aspjk4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': []}
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
        tmp1 = tl.load(in_ptr1 + (200704 + (3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7skeaperzxp4stp54gemcjjvo7jravvgxp2giiods4cwi2uven.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sfj4owvfd5sqozj7cxljgfgwir66ww7lfhglulmum63nuj5qjk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcyrlyqt4sf32dvixlk6m4vjecplngo6cjifqrd5m2yem75ldyl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (200704 + x4 + (401408*x2)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/b4/cb4ww3utguos25gictexwbxythqf46by5pcq4dlpoudrwwuguygz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []}
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
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (100352 + (3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4okafdvsra5ykftkqpbzhisipfwghmahbo4xwchmmsc4bg27r2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (100352 + x4 + (401408*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
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


# kernel path: /tmp/torchinductor_youkaichao/nx/cnx2awdvifjxurjvurnbktvqksveg32ws5cdouozbsho2u7vlbsx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': []}
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
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjl2ij3c7syb25mphysnndbxrtggkxt2hctm6z3ddnwfztx46k4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4 + (401408*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
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


# kernel path: /tmp/torchinductor_youkaichao/af/caf65ah3ox2bhf3f73bxoe3rfehe2v7ktehvigbxmjk4nyc6yd6t.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_82', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 32, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (100352*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 64, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-100352) + x4 + (100352*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 96, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-200704) + x4 + (100352*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 128, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_out_ptr0 + (x3), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wt/cwt4wafko34hvsx6pmbzoc4imni7hrd3o2rpa6dunaru5dve4ijc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhkyq44ypievh7pzesqcgvtwxdiwcoz6jk2sbaplnbfv7ufvp6l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_84', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ll/cllxebbkgnj4ymhpred7uhfvdeah3c76ptsgxjhnknzmmdjmjnn5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_85', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvoey4qtqncjsultjegwnoclzlmzggyem6vclyk3pofutu4eexs.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_86', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 3.985969387755102e-05
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


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3es7laj4bdtcryt755kbs5is3l3dmuprx2gkeer6e6g5uno6ea.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_87', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/6s/c6s3id327vgfx7varbtbgbhyhkjsjtl4uyztxrcb7mlt4s3mgsde.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 3.985969387755102e-05
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


# kernel path: /tmp/torchinductor_youkaichao/ny/cnyx5n7vqwnkfdwvvsmfzg2gbgydrqnpukdbqfjhvwrbrlicj24m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zs/czsxueu2terhgpjcgypmgkxpkk4pjjudqmwcgnc2yscy3lthxq7v.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
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


# kernel path: /tmp/torchinductor_youkaichao/qg/cqg5yh2a5d7yytwieeclbp47nl4rh7bcyzomzcdpofjex4r7mjoo.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136) % 32
    x3 = (xindex // 100352)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(56, 2 + x0))))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), None)
    tmp25 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(56, 2 + x0))))), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), None)
    tmp42 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(56, 2 + x0))))), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (301056 + (56*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (401408*x3) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(56, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(56, 2 + x0)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (-1) + x0))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.math.max(0, (-1) + x0))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.math.max(0, (-1) + x1))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.math.max(0, (-1) + x1))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (x6), tmp58, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cunvrgz4hwtcogd4cmbsted3pvy5z76sr3x2pau2yq6bk3uqgsrz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_92', 'mutated_arg_names': []}
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
        tmp1 = tl.load(in_ptr1 + (100352 + (3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nt24oyz4mooizs76634lbvl5eajzgwwvkwyi2dnyzn6e2jwhnv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (100352 + x4 + (401408*x2)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/to/cton4vbtkbtnq4jtc2ccifzfafjgtcjeaspcthuebesvhaqkazml.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_94', 'mutated_arg_names': []}
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
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zs/czsxjvvqv2nhbyxtcaqkm6gshe4s4mhsbmwgf4vqmxrbhaajtvtn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_95', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x4 + (401408*x2)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvzrr4hl6bvhurlqq4ugenkg3g3lbt7amcmrigbdkovdlnwrquw.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 32, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (100352*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 64, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-100352) + x4 + (100352*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 96, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-200704) + x4 + (100352*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 128, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-301056) + x4 + (100352*x2)), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp33, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/ctepphdsnmbjiqkemnv37w5erysfryte5dv2hmt4ve4ast4sz3dw.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_97', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/at/catujs2l3m446rne2jggs5a4jar2v4cyvro4vpwgyfglwhz3dpc3.py
# Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]

triton_poi_fused_add_max_pool2d_with_indices_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = (xindex // 112) % 112
    x2 = (xindex // 12544)
    x3 = xindex % 12544
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp7 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp19 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp31 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
    tmp2 = x3
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (x1 // 2))
    tmp10 = tl.math.min(56, 1 + ((1 + x1) // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp13 = tl.math.min(56, 1 + ((1 + x0) // 2))
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


# kernel path: /tmp/torchinductor_youkaichao/c2/cc273f5q3tzsswpeuw3l7pnqtpbh6ypvnhspe6qniop52elpn5iz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
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
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ug/cugwgc65fhexht2au6fxljyriqxcwdni34jv5s56lrbnyyznl7bl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7d65vbg46dpfa5qcm3nmlyercljzbeu2yegbazdwigm3kmech6n.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtdywm7ele4x52dkjcd7uyi2iaajzqulgo7ir4sikh2hqp23h5x.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_102', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
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
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_513, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, getitem_10, convolution_2, squeeze_7, getitem_17, convolution_3, squeeze_10, getitem_24, convolution_4, squeeze_13, getitem_31, cat, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, getitem_42, convolution_8, squeeze_25, add_46, convolution_9, squeeze_28, add_52, convolution_10, squeeze_31, cat_1, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, getitem_72, convolution_13, squeeze_40, add_74, convolution_14, squeeze_43, add_80, convolution_15, squeeze_46, cat_2, convolution_16, squeeze_49, relu_15, convolution_17, squeeze_52, getitem_102, convolution_18, squeeze_55, getitem_109, convolution_19, squeeze_58, getitem_116, convolution_20, squeeze_61, getitem_123, cat_3, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, getitem_134, convolution_24, squeeze_73, add_133, convolution_25, squeeze_76, add_139, convolution_26, squeeze_79, cat_4, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, getitem_164, convolution_29, squeeze_88, add_161, convolution_30, squeeze_91, add_167, convolution_31, squeeze_94, cat_5, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, getitem_194, convolution_34, squeeze_103, add_189, convolution_35, squeeze_106, add_195, convolution_36, squeeze_109, cat_6, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, getitem_224, convolution_39, squeeze_118, getitem_231, convolution_40, squeeze_121, getitem_238, convolution_41, squeeze_124, getitem_245, cat_7, convolution_42, squeeze_127, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, getitem_256, convolution_45, squeeze_136, add_248, convolution_46, squeeze_139, add_254, convolution_47, squeeze_142, cat_8, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, getitem_286, convolution_50, squeeze_151, add_276, convolution_51, squeeze_154, add_282, convolution_52, squeeze_157, cat_9, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, getitem_316, convolution_55, squeeze_166, add_304, convolution_56, squeeze_169, add_310, convolution_57, squeeze_172, cat_10, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, getitem_346, convolution_60, squeeze_181, add_332, convolution_61, squeeze_184, add_338, convolution_62, squeeze_187, cat_11, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, getitem_376, convolution_65, squeeze_196, add_360, convolution_66, squeeze_199, add_366, convolution_67, squeeze_202, cat_12, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, getitem_406, convolution_70, squeeze_211, getitem_413, convolution_71, squeeze_214, getitem_420, convolution_72, squeeze_217, getitem_427, cat_13, convolution_73, squeeze_220, convolution_74, squeeze_223, relu_70, convolution_75, squeeze_226, getitem_438, convolution_76, squeeze_229, add_419, convolution_77, squeeze_232, add_425, convolution_78, squeeze_235, cat_14, convolution_79, squeeze_238, relu_75, convolution_80, squeeze_241, getitem_468, convolution_81, squeeze_244, add_447, convolution_82, squeeze_247, add_453, convolution_83, squeeze_250, cat_15, convolution_84, squeeze_253, view, permute_1, le, unsqueeze_342, le_1, unsqueeze_354, le_2, unsqueeze_366, le_3, unsqueeze_378, le_4, unsqueeze_390, unsqueeze_402, le_6, unsqueeze_414, le_7, unsqueeze_426, le_8, unsqueeze_438, le_9, unsqueeze_450, unsqueeze_462, unsqueeze_474, le_11, unsqueeze_486, le_12, unsqueeze_498, le_13, unsqueeze_510, le_14, unsqueeze_522, unsqueeze_534, le_16, unsqueeze_546, le_17, unsqueeze_558, le_18, unsqueeze_570, le_19, unsqueeze_582, unsqueeze_594, le_21, unsqueeze_606, le_22, unsqueeze_618, le_23, unsqueeze_630, le_24, unsqueeze_642, unsqueeze_654, le_26, unsqueeze_666, le_27, unsqueeze_678, le_28, unsqueeze_690, le_29, unsqueeze_702, unsqueeze_714, le_31, unsqueeze_726, le_32, unsqueeze_738, le_33, unsqueeze_750, le_34, unsqueeze_762, unsqueeze_774, le_36, unsqueeze_786, le_37, unsqueeze_798, le_38, unsqueeze_810, le_39, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_41, unsqueeze_858, le_42, unsqueeze_870, le_43, unsqueeze_882, le_44, unsqueeze_894, unsqueeze_906, le_46, unsqueeze_918, le_47, unsqueeze_930, le_48, unsqueeze_942, le_49, unsqueeze_954, unsqueeze_966, le_51, unsqueeze_978, le_52, unsqueeze_990, le_53, unsqueeze_1002, le_54, unsqueeze_1014, unsqueeze_1026, le_56, unsqueeze_1038, le_57, unsqueeze_1050, le_58, unsqueeze_1062, le_59, unsqueeze_1074, unsqueeze_1086, unsqueeze_1098, le_61, unsqueeze_1110, le_62, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, le_66, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, le_71, unsqueeze_1230, le_72, unsqueeze_1242, le_73, unsqueeze_1254, le_74, unsqueeze_1266, unsqueeze_1278, unsqueeze_1290, le_76, unsqueeze_1302, le_77, unsqueeze_1314, le_78, unsqueeze_1326, le_79, unsqueeze_1338, unsqueeze_1350, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_10, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_13, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_14, (32, ), (1, ))
    assert_size_stride(primals_16, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_22, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_26, (32, ), (1, ))
    assert_size_stride(primals_28, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_31, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_32, (32, ), (1, ))
    assert_size_stride(primals_34, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_40, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_41, (32, ), (1, ))
    assert_size_stride(primals_43, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_46, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_49, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_52, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_55, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_58, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_61, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_64, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_67, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_70, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_73, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_76, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_79, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_82, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_85, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_88, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_91, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_92, (64, ), (1, ))
    assert_size_stride(primals_94, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_97, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_100, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_103, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_104, (64, ), (1, ))
    assert_size_stride(primals_106, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_107, (64, ), (1, ))
    assert_size_stride(primals_109, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_110, (64, ), (1, ))
    assert_size_stride(primals_112, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_115, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_118, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_121, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_124, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_127, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_133, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_136, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_139, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_142, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_145, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_148, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_151, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_154, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_157, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_160, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_163, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_166, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_169, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_172, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_175, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_178, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_181, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_184, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_187, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_190, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_193, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_196, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_197, (128, ), (1, ))
    assert_size_stride(primals_199, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_202, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_203, (128, ), (1, ))
    assert_size_stride(primals_205, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_208, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_209, (1024, ), (1, ))
    assert_size_stride(primals_211, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_214, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_217, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_220, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_221, (2048, ), (1, ))
    assert_size_stride(primals_223, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_226, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_229, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_232, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_235, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_238, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_239, (2048, ), (1, ))
    assert_size_stride(primals_241, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_244, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_245, (256, ), (1, ))
    assert_size_stride(primals_247, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_250, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_253, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_254, (2048, ), (1, ))
    assert_size_stride(primals_513, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(getitem_2, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(getitem_3, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_1, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(squeeze_4, (128, ), (1, ))
    assert_size_stride(getitem_10, (8, 32, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_2, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_7, (32, ), (1, ))
    assert_size_stride(getitem_17, (8, 32, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_3, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_10, (32, ), (1, ))
    assert_size_stride(getitem_24, (8, 32, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_4, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_13, (32, ), (1, ))
    assert_size_stride(getitem_31, (8, 32, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(cat, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_5, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_16, (256, ), (1, ))
    assert_size_stride(convolution_6, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(relu_5, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_7, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(squeeze_22, (128, ), (1, ))
    assert_size_stride(getitem_42, (8, 32, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_8, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_25, (32, ), (1, ))
    assert_size_stride(add_46, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(convolution_9, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_28, (32, ), (1, ))
    assert_size_stride(add_52, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(convolution_10, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_31, (32, ), (1, ))
    assert_size_stride(cat_1, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_11, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(relu_10, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_12, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(getitem_72, (8, 32, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_13, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_40, (32, ), (1, ))
    assert_size_stride(add_74, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(convolution_14, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_43, (32, ), (1, ))
    assert_size_stride(add_80, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(convolution_15, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_46, (32, ), (1, ))
    assert_size_stride(cat_2, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(convolution_16, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_49, (256, ), (1, ))
    assert_size_stride(relu_15, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_17, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_52, (256, ), (1, ))
    assert_size_stride(getitem_102, (8, 64, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_18, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_55, (64, ), (1, ))
    assert_size_stride(getitem_109, (8, 64, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_19, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_58, (64, ), (1, ))
    assert_size_stride(getitem_116, (8, 64, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_20, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_61, (64, ), (1, ))
    assert_size_stride(getitem_123, (8, 64, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(cat_3, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_21, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_64, (512, ), (1, ))
    assert_size_stride(convolution_22, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_67, (512, ), (1, ))
    assert_size_stride(relu_20, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_23, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(squeeze_70, (256, ), (1, ))
    assert_size_stride(getitem_134, (8, 64, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_24, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_73, (64, ), (1, ))
    assert_size_stride(add_133, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(convolution_25, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_76, (64, ), (1, ))
    assert_size_stride(add_139, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(convolution_26, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_79, (64, ), (1, ))
    assert_size_stride(cat_4, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_27, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_82, (512, ), (1, ))
    assert_size_stride(relu_25, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_28, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(getitem_164, (8, 64, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_29, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_88, (64, ), (1, ))
    assert_size_stride(add_161, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(convolution_30, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_91, (64, ), (1, ))
    assert_size_stride(add_167, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(convolution_31, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_94, (64, ), (1, ))
    assert_size_stride(cat_5, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_32, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_97, (512, ), (1, ))
    assert_size_stride(relu_30, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_33, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(squeeze_100, (256, ), (1, ))
    assert_size_stride(getitem_194, (8, 64, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_34, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_103, (64, ), (1, ))
    assert_size_stride(add_189, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(convolution_35, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_106, (64, ), (1, ))
    assert_size_stride(add_195, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(convolution_36, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_109, (64, ), (1, ))
    assert_size_stride(cat_6, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(convolution_37, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_112, (512, ), (1, ))
    assert_size_stride(relu_35, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_38, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_115, (512, ), (1, ))
    assert_size_stride(getitem_224, (8, 128, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_39, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_118, (128, ), (1, ))
    assert_size_stride(getitem_231, (8, 128, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_40, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_121, (128, ), (1, ))
    assert_size_stride(getitem_238, (8, 128, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_41, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_124, (128, ), (1, ))
    assert_size_stride(getitem_245, (8, 128, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(cat_7, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_42, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_127, (1024, ), (1, ))
    assert_size_stride(convolution_43, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_130, (1024, ), (1, ))
    assert_size_stride(relu_40, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_44, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(squeeze_133, (512, ), (1, ))
    assert_size_stride(getitem_256, (8, 128, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_45, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_136, (128, ), (1, ))
    assert_size_stride(add_248, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_46, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_139, (128, ), (1, ))
    assert_size_stride(add_254, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_47, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_142, (128, ), (1, ))
    assert_size_stride(cat_8, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_48, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_145, (1024, ), (1, ))
    assert_size_stride(relu_45, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_49, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(squeeze_148, (512, ), (1, ))
    assert_size_stride(getitem_286, (8, 128, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_50, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_151, (128, ), (1, ))
    assert_size_stride(add_276, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_51, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_154, (128, ), (1, ))
    assert_size_stride(add_282, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_52, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_157, (128, ), (1, ))
    assert_size_stride(cat_9, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_53, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_160, (1024, ), (1, ))
    assert_size_stride(relu_50, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_54, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(squeeze_163, (512, ), (1, ))
    assert_size_stride(getitem_316, (8, 128, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_55, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_166, (128, ), (1, ))
    assert_size_stride(add_304, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_56, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_169, (128, ), (1, ))
    assert_size_stride(add_310, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_57, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_172, (128, ), (1, ))
    assert_size_stride(cat_10, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_58, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_175, (1024, ), (1, ))
    assert_size_stride(relu_55, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_59, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(squeeze_178, (512, ), (1, ))
    assert_size_stride(getitem_346, (8, 128, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_60, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_181, (128, ), (1, ))
    assert_size_stride(add_332, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_61, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_184, (128, ), (1, ))
    assert_size_stride(add_338, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_62, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_187, (128, ), (1, ))
    assert_size_stride(cat_11, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_63, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_190, (1024, ), (1, ))
    assert_size_stride(relu_60, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_64, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(squeeze_193, (512, ), (1, ))
    assert_size_stride(getitem_376, (8, 128, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_65, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_196, (128, ), (1, ))
    assert_size_stride(add_360, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_66, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_199, (128, ), (1, ))
    assert_size_stride(add_366, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(convolution_67, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_202, (128, ), (1, ))
    assert_size_stride(cat_12, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(convolution_68, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_205, (1024, ), (1, ))
    assert_size_stride(relu_65, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_69, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_208, (1024, ), (1, ))
    assert_size_stride(getitem_406, (8, 256, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_70, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_211, (256, ), (1, ))
    assert_size_stride(getitem_413, (8, 256, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_71, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_214, (256, ), (1, ))
    assert_size_stride(getitem_420, (8, 256, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_72, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_217, (256, ), (1, ))
    assert_size_stride(getitem_427, (8, 256, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(cat_13, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_73, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(squeeze_220, (2048, ), (1, ))
    assert_size_stride(convolution_74, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(squeeze_223, (2048, ), (1, ))
    assert_size_stride(relu_70, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(convolution_75, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(squeeze_226, (1024, ), (1, ))
    assert_size_stride(getitem_438, (8, 256, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_76, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_229, (256, ), (1, ))
    assert_size_stride(add_419, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(convolution_77, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_232, (256, ), (1, ))
    assert_size_stride(add_425, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(convolution_78, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_235, (256, ), (1, ))
    assert_size_stride(cat_14, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_79, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(squeeze_238, (2048, ), (1, ))
    assert_size_stride(relu_75, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(convolution_80, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(squeeze_241, (1024, ), (1, ))
    assert_size_stride(getitem_468, (8, 256, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_81, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_244, (256, ), (1, ))
    assert_size_stride(add_447, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(convolution_82, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_247, (256, ), (1, ))
    assert_size_stride(add_453, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(convolution_83, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(squeeze_250, (256, ), (1, ))
    assert_size_stride(cat_15, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(convolution_84, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(squeeze_253, (2048, ), (1, ))
    assert_size_stride(view, (8, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(unsqueeze_342, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_1, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_354, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_2, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_366, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_3, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_378, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_4, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(unsqueeze_390, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_6, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_414, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_7, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_426, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_8, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_438, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_9, (8, 1024, 7, 7), (50176, 49, 7, 1))
    assert_size_stride(unsqueeze_450, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_11, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_486, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_12, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_498, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_13, (8, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(unsqueeze_510, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_14, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(unsqueeze_522, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_16, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_546, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_17, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_558, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_18, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_570, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_19, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(unsqueeze_582, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_594, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_21, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_606, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_22, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_618, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_23, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_630, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_24, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(unsqueeze_642, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_26, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_666, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_27, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_678, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_28, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_690, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_29, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(unsqueeze_702, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_31, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_726, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_32, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_738, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_33, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_750, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_34, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(unsqueeze_762, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_774, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_36, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_786, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_37, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_798, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_38, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_810, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_39, (8, 512, 14, 14), (100352, 196, 14, 1))
    assert_size_stride(unsqueeze_822, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_41, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_858, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_42, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_870, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_43, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(unsqueeze_882, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_44, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(unsqueeze_894, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_906, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_46, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_918, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_47, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_930, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_48, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_942, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_49, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(unsqueeze_954, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_966, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_51, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_978, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_52, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_990, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_53, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_1002, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_54, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(unsqueeze_1014, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1026, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_56, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_1038, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_57, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_1050, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_58, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_1062, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_59, (8, 256, 28, 28), (200704, 784, 28, 1))
    assert_size_stride(unsqueeze_1074, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1086, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1098, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_61, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_1110, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_62, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_1122, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_63, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(unsqueeze_1134, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_64, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(unsqueeze_1146, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1158, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_66, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1170, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_67, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1182, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_68, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1194, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_69, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(unsqueeze_1206, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1218, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_71, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1230, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_72, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1242, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_73, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1254, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_74, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(unsqueeze_1266, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1278, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1290, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_76, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1302, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_77, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1314, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_78, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(unsqueeze_1326, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_79, (8, 128, 56, 56), (401408, 3136, 56, 1))
    assert_size_stride(unsqueeze_1338, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1350, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view, out=buf1)
        del view
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf4 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf5 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_84, unsqueeze_342, squeeze_253, buf3, buf4, buf5, 2048, 392, grid=grid(2048), stream=stream0)
        buf6 = empty((8, 2048, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2.run(le, buf0, convolution_84, unsqueeze_342, buf4, squeeze_253, buf3, primals_254, buf6, 802816, grid=grid(802816), stream=stream0)
        del convolution_84
        del primals_254
        del squeeze_253
        del unsqueeze_342
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf7 = aten.convolution_backward(buf6, cat_15, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_15
        del primals_253
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        buf10 = empty((256, ), device='cuda', dtype=torch.float32)
        buf11 = empty((256, ), device='cuda', dtype=torch.float32)
        buf12 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(le_1, buf8, convolution_83, unsqueeze_354, squeeze_250, buf10, buf11, buf12, 256, 392, grid=grid(256), stream=stream0)
        buf13 = empty((8, 256, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(le_1, buf8, convolution_83, unsqueeze_354, buf11, squeeze_250, buf10, primals_251, buf13, 100352, grid=grid(100352), stream=stream0)
        del convolution_83
        del le_1
        del primals_251
        del squeeze_250
        del unsqueeze_354
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf14 = aten.convolution_backward(buf13, add_453, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_453
        del primals_250
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = buf11; del buf11  # reuse
        buf18 = empty((256, ), device='cuda', dtype=torch.float32)
        buf20 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_5.run(le_2, buf8, buf15, convolution_82, unsqueeze_366, squeeze_247, buf17, buf18, buf20, 256, 392, grid=grid(256), stream=stream0)
        buf19 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_6.run(le_2, buf8, buf15, convolution_82, unsqueeze_366, buf18, squeeze_247, buf17, primals_248, buf19, 100352, grid=grid(100352), stream=stream0)
        del convolution_82
        del le_2
        del primals_248
        del squeeze_247
        del unsqueeze_366
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf21 = aten.convolution_backward(buf19, add_447, primals_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_447
        del primals_247
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = buf18; del buf18  # reuse
        buf25 = empty((256, ), device='cuda', dtype=torch.float32)
        buf27 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(le_3, buf8, buf22, convolution_81, unsqueeze_378, squeeze_244, buf24, buf25, buf27, 256, 392, grid=grid(256), stream=stream0)
        buf26 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_8.run(le_3, buf8, buf22, convolution_81, unsqueeze_378, buf25, squeeze_244, buf24, primals_245, buf26, 100352, grid=grid(100352), stream=stream0)
        del convolution_81
        del le_3
        del primals_245
        del squeeze_244
        del unsqueeze_378
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf28 = aten.convolution_backward(buf26, getitem_468, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf26
        del getitem_468
        del primals_244
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_9.run(buf31, le_4, buf29, buf22, buf15, 401408, grid=grid(401408), stream=stream0)
        del buf15
        del buf22
        del le_4
        buf32 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf33 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf34 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_10.run(buf31, convolution_80, unsqueeze_390, squeeze_241, buf32, buf33, buf34, 1024, 392, grid=grid(1024), stream=stream0)
        buf35 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_11.run(buf35, convolution_80, unsqueeze_390, buf33, squeeze_241, buf32, primals_242, 401408, grid=grid(401408), stream=stream0)
        del convolution_80
        del primals_242
        del squeeze_241
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf36 = aten.convolution_backward(buf35, relu_75, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf35
        del primals_241
        buf37 = buf36[0]
        buf38 = buf36[1]
        del buf36
        buf39 = buf4; del buf4  # reuse
        buf40 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf42 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_12.run(relu_75, le, buf0, buf37, convolution_79, unsqueeze_402, squeeze_238, buf39, buf40, buf42, 2048, 392, grid=grid(2048), stream=stream0)
        buf41 = buf6; del buf6  # reuse
        buf43 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_13.run(buf43, relu_75, le, buf0, buf37, convolution_79, unsqueeze_402, buf40, squeeze_238, buf39, primals_239, 802816, grid=grid(802816), stream=stream0)
        del convolution_79
        del primals_239
        del squeeze_238
        del unsqueeze_402
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf44 = aten.convolution_backward(buf43, cat_14, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_14
        del primals_238
        buf45 = buf44[0]
        buf46 = buf44[1]
        del buf44
        buf47 = buf25; del buf25  # reuse
        buf48 = empty((256, ), device='cuda', dtype=torch.float32)
        buf49 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(le_6, buf45, convolution_78, unsqueeze_414, squeeze_235, buf47, buf48, buf49, 256, 392, grid=grid(256), stream=stream0)
        buf50 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(le_6, buf45, convolution_78, unsqueeze_414, buf48, squeeze_235, buf47, primals_236, buf50, 100352, grid=grid(100352), stream=stream0)
        del convolution_78
        del le_6
        del primals_236
        del squeeze_235
        del unsqueeze_414
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf51 = aten.convolution_backward(buf50, add_425, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_425
        del primals_235
        buf52 = buf51[0]
        buf53 = buf51[1]
        del buf51
        buf54 = buf48; del buf48  # reuse
        buf55 = empty((256, ), device='cuda', dtype=torch.float32)
        buf57 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_5.run(le_7, buf45, buf52, convolution_77, unsqueeze_426, squeeze_232, buf54, buf55, buf57, 256, 392, grid=grid(256), stream=stream0)
        buf56 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_6.run(le_7, buf45, buf52, convolution_77, unsqueeze_426, buf55, squeeze_232, buf54, primals_233, buf56, 100352, grid=grid(100352), stream=stream0)
        del convolution_77
        del le_7
        del primals_233
        del squeeze_232
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf58 = aten.convolution_backward(buf56, add_419, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_419
        del primals_232
        buf59 = buf58[0]
        buf60 = buf58[1]
        del buf58
        buf61 = buf55; del buf55  # reuse
        buf62 = empty((256, ), device='cuda', dtype=torch.float32)
        buf64 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(le_8, buf45, buf59, convolution_76, unsqueeze_438, squeeze_229, buf61, buf62, buf64, 256, 392, grid=grid(256), stream=stream0)
        buf63 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_8.run(le_8, buf45, buf59, convolution_76, unsqueeze_438, buf62, squeeze_229, buf61, primals_230, buf63, 100352, grid=grid(100352), stream=stream0)
        del convolution_76
        del le_8
        del primals_230
        del squeeze_229
        del unsqueeze_438
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf65 = aten.convolution_backward(buf63, getitem_438, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf63
        del getitem_438
        del primals_229
        buf66 = buf65[0]
        buf67 = buf65[1]
        del buf65
        buf68 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_9.run(buf68, le_9, buf66, buf59, buf52, 401408, grid=grid(401408), stream=stream0)
        del buf52
        del buf59
        del le_9
        buf69 = buf33; del buf33  # reuse
        buf70 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf71 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_10.run(buf68, convolution_75, unsqueeze_450, squeeze_226, buf69, buf70, buf71, 1024, 392, grid=grid(1024), stream=stream0)
        buf72 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_11.run(buf72, convolution_75, unsqueeze_450, buf70, squeeze_226, buf69, primals_227, 401408, grid=grid(401408), stream=stream0)
        del convolution_75
        del primals_227
        del squeeze_226
        del unsqueeze_450
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf73 = aten.convolution_backward(buf72, relu_70, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_226
        buf74 = buf73[0]
        buf75 = buf73[1]
        del buf73
        buf76 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
        triton_poi_fused_add_div_threshold_backward_14.run(buf76, relu_70, relu_75, le, buf0, buf74, 802816, grid=grid(802816), stream=stream0)
        del buf0
        del le
        del relu_70
        del relu_75
        buf77 = buf40; del buf40  # reuse
        buf78 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf84 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf79 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf85 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_15.run(buf76, convolution_74, unsqueeze_462, convolution_73, unsqueeze_474, squeeze_223, squeeze_220, buf77, buf78, buf84, buf79, buf85, 2048, 392, grid=grid(2048), stream=stream0)
        buf80 = buf74; del buf74  # reuse
        buf86 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_16.run(buf76, convolution_74, unsqueeze_462, buf78, squeeze_223, buf77, primals_224, convolution_73, unsqueeze_474, buf84, squeeze_220, primals_221, buf80, buf86, 802816, grid=grid(802816), stream=stream0)
        del buf76
        del buf78
        del buf84
        del convolution_73
        del convolution_74
        del primals_221
        del primals_224
        del squeeze_220
        del squeeze_223
        del unsqueeze_462
        del unsqueeze_474
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf81 = aten.convolution_backward(buf80, relu_65, primals_223, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf80
        del primals_223
        buf82 = buf81[0]
        buf83 = buf81[1]
        del buf81
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf87 = aten.convolution_backward(buf86, cat_13, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf86
        del cat_13
        del primals_220
        buf88 = buf87[0]
        buf89 = buf87[1]
        del buf87
        buf90 = reinterpret_tensor(buf72, (8, 256, 14, 14), (50176, 196, 14, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_17.run(buf88, buf90, 401408, grid=grid(401408), stream=stream0)
        buf91 = buf62; del buf62  # reuse
        buf92 = empty((256, ), device='cuda', dtype=torch.float32)
        buf93 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(le_11, buf88, convolution_72, unsqueeze_486, squeeze_217, buf91, buf92, buf93, 256, 392, grid=grid(256), stream=stream0)
        buf94 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(le_11, buf88, convolution_72, unsqueeze_486, buf92, squeeze_217, buf91, primals_218, buf94, 100352, grid=grid(100352), stream=stream0)
        del convolution_72
        del le_11
        del primals_218
        del squeeze_217
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf95 = aten.convolution_backward(buf94, getitem_420, primals_217, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del getitem_420
        del primals_217
        buf96 = buf95[0]
        buf97 = buf95[1]
        del buf95
        buf98 = buf92; del buf92  # reuse
        buf99 = empty((256, ), device='cuda', dtype=torch.float32)
        buf100 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(le_12, buf88, convolution_71, unsqueeze_498, squeeze_214, buf98, buf99, buf100, 256, 392, grid=grid(256), stream=stream0)
        buf101 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19.run(le_12, buf88, convolution_71, unsqueeze_498, buf99, squeeze_214, buf98, primals_215, buf101, 100352, grid=grid(100352), stream=stream0)
        del convolution_71
        del le_12
        del primals_215
        del squeeze_214
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf102 = aten.convolution_backward(buf101, getitem_413, primals_214, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del getitem_413
        del primals_214
        buf103 = buf102[0]
        buf104 = buf102[1]
        del buf102
        buf105 = buf99; del buf99  # reuse
        buf106 = empty((256, ), device='cuda', dtype=torch.float32)
        buf107 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(le_13, buf88, convolution_70, unsqueeze_510, squeeze_211, buf105, buf106, buf107, 256, 392, grid=grid(256), stream=stream0)
        buf108 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(le_13, buf88, convolution_70, unsqueeze_510, buf106, squeeze_211, buf105, primals_212, buf108, 100352, grid=grid(100352), stream=stream0)
        del buf88
        del convolution_70
        del le_13
        del primals_212
        del squeeze_211
        del unsqueeze_510
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf109 = aten.convolution_backward(buf108, getitem_406, primals_211, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf108
        del getitem_406
        del primals_211
        buf110 = buf109[0]
        buf111 = buf109[1]
        del buf109
        buf112 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_22.run(le_14, buf110, buf103, buf96, buf90, buf112, 1605632, grid=grid(1605632), stream=stream0)
        del buf103
        del buf110
        del buf90
        del le_14
        buf113 = buf70; del buf70  # reuse
        buf114 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf115 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_23.run(buf112, convolution_69, unsqueeze_522, squeeze_208, buf113, buf114, buf115, 1024, 1568, grid=grid(1024), stream=stream0)
        buf116 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_24.run(buf116, convolution_69, unsqueeze_522, buf114, squeeze_208, buf113, primals_209, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_69
        del primals_209
        del squeeze_208
        del unsqueeze_522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf117 = aten.convolution_backward(buf116, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_208
        buf118 = buf117[0]
        buf119 = buf117[1]
        del buf117
        buf120 = buf114; del buf114  # reuse
        buf121 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf123 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_25.run(relu_65, buf82, buf118, convolution_68, unsqueeze_534, squeeze_205, buf120, buf121, buf123, 1024, 1568, grid=grid(1024), stream=stream0)
        buf122 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_26.run(relu_65, buf82, buf118, convolution_68, unsqueeze_534, buf121, squeeze_205, buf120, primals_206, buf122, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_68
        del primals_206
        del squeeze_205
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf124 = aten.convolution_backward(buf122, cat_12, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf122
        del cat_12
        del primals_205
        buf125 = buf124[0]
        buf126 = buf124[1]
        del buf124
        buf127 = empty((128, ), device='cuda', dtype=torch.float32)
        buf128 = empty((128, ), device='cuda', dtype=torch.float32)
        buf129 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(le_16, buf125, convolution_67, unsqueeze_546, squeeze_202, buf127, buf128, buf129, 128, 1568, grid=grid(128), stream=stream0)
        buf130 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(le_16, buf125, convolution_67, unsqueeze_546, buf128, squeeze_202, buf127, primals_203, buf130, 200704, grid=grid(200704), stream=stream0)
        del convolution_67
        del le_16
        del primals_203
        del squeeze_202
        del unsqueeze_546
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf131 = aten.convolution_backward(buf130, add_366, primals_202, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_366
        del primals_202
        buf132 = buf131[0]
        buf133 = buf131[1]
        del buf131
        buf134 = buf128; del buf128  # reuse
        buf135 = empty((128, ), device='cuda', dtype=torch.float32)
        buf137 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_29.run(le_17, buf125, buf132, convolution_66, unsqueeze_558, squeeze_199, buf134, buf135, buf137, 128, 1568, grid=grid(128), stream=stream0)
        buf136 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_30.run(le_17, buf125, buf132, convolution_66, unsqueeze_558, buf135, squeeze_199, buf134, primals_200, buf136, 200704, grid=grid(200704), stream=stream0)
        del convolution_66
        del le_17
        del primals_200
        del squeeze_199
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf138 = aten.convolution_backward(buf136, add_360, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_360
        del primals_199
        buf139 = buf138[0]
        buf140 = buf138[1]
        del buf138
        buf141 = buf135; del buf135  # reuse
        buf142 = empty((128, ), device='cuda', dtype=torch.float32)
        buf144 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_31.run(le_18, buf125, buf139, convolution_65, unsqueeze_570, squeeze_196, buf141, buf142, buf144, 128, 1568, grid=grid(128), stream=stream0)
        buf143 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32.run(le_18, buf125, buf139, convolution_65, unsqueeze_570, buf142, squeeze_196, buf141, primals_197, buf143, 200704, grid=grid(200704), stream=stream0)
        del convolution_65
        del le_18
        del primals_197
        del squeeze_196
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf145 = aten.convolution_backward(buf143, getitem_376, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf143
        del getitem_376
        del primals_196
        buf146 = buf145[0]
        buf147 = buf145[1]
        del buf145
        buf148 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_33.run(buf148, le_19, buf146, buf139, buf132, 802816, grid=grid(802816), stream=stream0)
        del buf132
        del buf139
        del le_19
        buf149 = empty((512, ), device='cuda', dtype=torch.float32)
        buf150 = empty((512, ), device='cuda', dtype=torch.float32)
        buf151 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_34.run(buf148, convolution_64, unsqueeze_582, squeeze_193, buf149, buf150, buf151, 512, 1568, grid=grid(512), stream=stream0)
        buf152 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_35.run(buf152, convolution_64, unsqueeze_582, buf150, squeeze_193, buf149, primals_194, 802816, grid=grid(802816), stream=stream0)
        del convolution_64
        del primals_194
        del squeeze_193
        del unsqueeze_582
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf153 = aten.convolution_backward(buf152, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf152
        del primals_193
        buf154 = buf153[0]
        buf155 = buf153[1]
        del buf153
        buf156 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_36.run(buf156, relu_60, relu_65, buf82, buf154, 1605632, grid=grid(1605632), stream=stream0)
        del buf154
        del relu_60
        del relu_65
        buf157 = buf121; del buf121  # reuse
        buf158 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf159 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_23.run(buf156, convolution_63, unsqueeze_594, squeeze_190, buf157, buf158, buf159, 1024, 1568, grid=grid(1024), stream=stream0)
        buf160 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_37.run(buf156, convolution_63, unsqueeze_594, buf158, squeeze_190, buf157, primals_191, buf160, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_63
        del primals_191
        del squeeze_190
        del unsqueeze_594
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf161 = aten.convolution_backward(buf160, cat_11, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_11
        del primals_190
        buf162 = buf161[0]
        buf163 = buf161[1]
        del buf161
        buf164 = buf142; del buf142  # reuse
        buf165 = empty((128, ), device='cuda', dtype=torch.float32)
        buf166 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(le_21, buf162, convolution_62, unsqueeze_606, squeeze_187, buf164, buf165, buf166, 128, 1568, grid=grid(128), stream=stream0)
        buf167 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(le_21, buf162, convolution_62, unsqueeze_606, buf165, squeeze_187, buf164, primals_188, buf167, 200704, grid=grid(200704), stream=stream0)
        del convolution_62
        del le_21
        del primals_188
        del squeeze_187
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf168 = aten.convolution_backward(buf167, add_338, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_338
        del primals_187
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = buf165; del buf165  # reuse
        buf172 = empty((128, ), device='cuda', dtype=torch.float32)
        buf174 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_29.run(le_22, buf162, buf169, convolution_61, unsqueeze_618, squeeze_184, buf171, buf172, buf174, 128, 1568, grid=grid(128), stream=stream0)
        buf173 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_30.run(le_22, buf162, buf169, convolution_61, unsqueeze_618, buf172, squeeze_184, buf171, primals_185, buf173, 200704, grid=grid(200704), stream=stream0)
        del convolution_61
        del le_22
        del primals_185
        del squeeze_184
        del unsqueeze_618
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf175 = aten.convolution_backward(buf173, add_332, primals_184, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_332
        del primals_184
        buf176 = buf175[0]
        buf177 = buf175[1]
        del buf175
        buf178 = buf172; del buf172  # reuse
        buf179 = empty((128, ), device='cuda', dtype=torch.float32)
        buf181 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_31.run(le_23, buf162, buf176, convolution_60, unsqueeze_630, squeeze_181, buf178, buf179, buf181, 128, 1568, grid=grid(128), stream=stream0)
        buf180 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32.run(le_23, buf162, buf176, convolution_60, unsqueeze_630, buf179, squeeze_181, buf178, primals_182, buf180, 200704, grid=grid(200704), stream=stream0)
        del convolution_60
        del le_23
        del primals_182
        del squeeze_181
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf182 = aten.convolution_backward(buf180, getitem_346, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf180
        del getitem_346
        del primals_181
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        buf185 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_33.run(buf185, le_24, buf183, buf176, buf169, 802816, grid=grid(802816), stream=stream0)
        del buf169
        del buf176
        del le_24
        buf186 = buf150; del buf150  # reuse
        buf187 = empty((512, ), device='cuda', dtype=torch.float32)
        buf188 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_34.run(buf185, convolution_59, unsqueeze_642, squeeze_178, buf186, buf187, buf188, 512, 1568, grid=grid(512), stream=stream0)
        buf189 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_35.run(buf189, convolution_59, unsqueeze_642, buf187, squeeze_178, buf186, primals_179, 802816, grid=grid(802816), stream=stream0)
        del convolution_59
        del primals_179
        del squeeze_178
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf190 = aten.convolution_backward(buf189, relu_55, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf189
        del primals_178
        buf191 = buf190[0]
        buf192 = buf190[1]
        del buf190
        buf193 = buf158; del buf158  # reuse
        buf194 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf196 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_25.run(relu_55, buf156, buf191, convolution_58, unsqueeze_654, squeeze_175, buf193, buf194, buf196, 1024, 1568, grid=grid(1024), stream=stream0)
        buf195 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_26.run(relu_55, buf156, buf191, convolution_58, unsqueeze_654, buf194, squeeze_175, buf193, primals_176, buf195, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_58
        del primals_176
        del squeeze_175
        del unsqueeze_654
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf197 = aten.convolution_backward(buf195, cat_10, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf195
        del cat_10
        del primals_175
        buf198 = buf197[0]
        buf199 = buf197[1]
        del buf197
        buf200 = buf179; del buf179  # reuse
        buf201 = empty((128, ), device='cuda', dtype=torch.float32)
        buf202 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(le_26, buf198, convolution_57, unsqueeze_666, squeeze_172, buf200, buf201, buf202, 128, 1568, grid=grid(128), stream=stream0)
        buf203 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(le_26, buf198, convolution_57, unsqueeze_666, buf201, squeeze_172, buf200, primals_173, buf203, 200704, grid=grid(200704), stream=stream0)
        del convolution_57
        del le_26
        del primals_173
        del squeeze_172
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf204 = aten.convolution_backward(buf203, add_310, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_310
        del primals_172
        buf205 = buf204[0]
        buf206 = buf204[1]
        del buf204
        buf207 = buf201; del buf201  # reuse
        buf208 = empty((128, ), device='cuda', dtype=torch.float32)
        buf210 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_29.run(le_27, buf198, buf205, convolution_56, unsqueeze_678, squeeze_169, buf207, buf208, buf210, 128, 1568, grid=grid(128), stream=stream0)
        buf209 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_30.run(le_27, buf198, buf205, convolution_56, unsqueeze_678, buf208, squeeze_169, buf207, primals_170, buf209, 200704, grid=grid(200704), stream=stream0)
        del convolution_56
        del le_27
        del primals_170
        del squeeze_169
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf211 = aten.convolution_backward(buf209, add_304, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_304
        del primals_169
        buf212 = buf211[0]
        buf213 = buf211[1]
        del buf211
        buf214 = buf208; del buf208  # reuse
        buf215 = empty((128, ), device='cuda', dtype=torch.float32)
        buf217 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_31.run(le_28, buf198, buf212, convolution_55, unsqueeze_690, squeeze_166, buf214, buf215, buf217, 128, 1568, grid=grid(128), stream=stream0)
        buf216 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32.run(le_28, buf198, buf212, convolution_55, unsqueeze_690, buf215, squeeze_166, buf214, primals_167, buf216, 200704, grid=grid(200704), stream=stream0)
        del convolution_55
        del le_28
        del primals_167
        del squeeze_166
        del unsqueeze_690
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf218 = aten.convolution_backward(buf216, getitem_316, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf216
        del getitem_316
        del primals_166
        buf219 = buf218[0]
        buf220 = buf218[1]
        del buf218
        buf221 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_33.run(buf221, le_29, buf219, buf212, buf205, 802816, grid=grid(802816), stream=stream0)
        del buf205
        del buf212
        del le_29
        buf222 = buf187; del buf187  # reuse
        buf223 = empty((512, ), device='cuda', dtype=torch.float32)
        buf224 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_34.run(buf221, convolution_54, unsqueeze_702, squeeze_163, buf222, buf223, buf224, 512, 1568, grid=grid(512), stream=stream0)
        buf225 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_35.run(buf225, convolution_54, unsqueeze_702, buf223, squeeze_163, buf222, primals_164, 802816, grid=grid(802816), stream=stream0)
        del convolution_54
        del primals_164
        del squeeze_163
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf226 = aten.convolution_backward(buf225, relu_50, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf225
        del primals_163
        buf227 = buf226[0]
        buf228 = buf226[1]
        del buf226
        buf229 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_38.run(buf229, relu_50, relu_55, buf191, buf227, 1605632, grid=grid(1605632), stream=stream0)
        del buf191
        del relu_50
        del relu_55
        buf230 = buf194; del buf194  # reuse
        buf231 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf232 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_23.run(buf229, convolution_53, unsqueeze_714, squeeze_160, buf230, buf231, buf232, 1024, 1568, grid=grid(1024), stream=stream0)
        buf233 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_37.run(buf229, convolution_53, unsqueeze_714, buf231, squeeze_160, buf230, primals_161, buf233, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_53
        del primals_161
        del squeeze_160
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf234 = aten.convolution_backward(buf233, cat_9, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_9
        del primals_160
        buf235 = buf234[0]
        buf236 = buf234[1]
        del buf234
        buf237 = buf215; del buf215  # reuse
        buf238 = empty((128, ), device='cuda', dtype=torch.float32)
        buf239 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(le_31, buf235, convolution_52, unsqueeze_726, squeeze_157, buf237, buf238, buf239, 128, 1568, grid=grid(128), stream=stream0)
        buf240 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(le_31, buf235, convolution_52, unsqueeze_726, buf238, squeeze_157, buf237, primals_158, buf240, 200704, grid=grid(200704), stream=stream0)
        del convolution_52
        del le_31
        del primals_158
        del squeeze_157
        del unsqueeze_726
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf241 = aten.convolution_backward(buf240, add_282, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_282
        del primals_157
        buf242 = buf241[0]
        buf243 = buf241[1]
        del buf241
        buf244 = buf238; del buf238  # reuse
        buf245 = empty((128, ), device='cuda', dtype=torch.float32)
        buf247 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_29.run(le_32, buf235, buf242, convolution_51, unsqueeze_738, squeeze_154, buf244, buf245, buf247, 128, 1568, grid=grid(128), stream=stream0)
        buf246 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_30.run(le_32, buf235, buf242, convolution_51, unsqueeze_738, buf245, squeeze_154, buf244, primals_155, buf246, 200704, grid=grid(200704), stream=stream0)
        del convolution_51
        del le_32
        del primals_155
        del squeeze_154
        del unsqueeze_738
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf248 = aten.convolution_backward(buf246, add_276, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_276
        del primals_154
        buf249 = buf248[0]
        buf250 = buf248[1]
        del buf248
        buf251 = buf245; del buf245  # reuse
        buf252 = empty((128, ), device='cuda', dtype=torch.float32)
        buf254 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_31.run(le_33, buf235, buf249, convolution_50, unsqueeze_750, squeeze_151, buf251, buf252, buf254, 128, 1568, grid=grid(128), stream=stream0)
        buf253 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32.run(le_33, buf235, buf249, convolution_50, unsqueeze_750, buf252, squeeze_151, buf251, primals_152, buf253, 200704, grid=grid(200704), stream=stream0)
        del convolution_50
        del le_33
        del primals_152
        del squeeze_151
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf255 = aten.convolution_backward(buf253, getitem_286, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf253
        del getitem_286
        del primals_151
        buf256 = buf255[0]
        buf257 = buf255[1]
        del buf255
        buf258 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_33.run(buf258, le_34, buf256, buf249, buf242, 802816, grid=grid(802816), stream=stream0)
        del buf242
        del buf249
        del le_34
        buf259 = buf223; del buf223  # reuse
        buf260 = empty((512, ), device='cuda', dtype=torch.float32)
        buf261 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_34.run(buf258, convolution_49, unsqueeze_762, squeeze_148, buf259, buf260, buf261, 512, 1568, grid=grid(512), stream=stream0)
        buf262 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_35.run(buf262, convolution_49, unsqueeze_762, buf260, squeeze_148, buf259, primals_149, 802816, grid=grid(802816), stream=stream0)
        del convolution_49
        del primals_149
        del squeeze_148
        del unsqueeze_762
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf263 = aten.convolution_backward(buf262, relu_45, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf262
        del primals_148
        buf264 = buf263[0]
        buf265 = buf263[1]
        del buf263
        buf266 = buf231; del buf231  # reuse
        buf267 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf269 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_25.run(relu_45, buf229, buf264, convolution_48, unsqueeze_774, squeeze_145, buf266, buf267, buf269, 1024, 1568, grid=grid(1024), stream=stream0)
        buf268 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_26.run(relu_45, buf229, buf264, convolution_48, unsqueeze_774, buf267, squeeze_145, buf266, primals_146, buf268, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_48
        del primals_146
        del squeeze_145
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf270 = aten.convolution_backward(buf268, cat_8, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf268
        del cat_8
        del primals_145
        buf271 = buf270[0]
        buf272 = buf270[1]
        del buf270
        buf273 = buf252; del buf252  # reuse
        buf274 = empty((128, ), device='cuda', dtype=torch.float32)
        buf275 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(le_36, buf271, convolution_47, unsqueeze_786, squeeze_142, buf273, buf274, buf275, 128, 1568, grid=grid(128), stream=stream0)
        buf276 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(le_36, buf271, convolution_47, unsqueeze_786, buf274, squeeze_142, buf273, primals_143, buf276, 200704, grid=grid(200704), stream=stream0)
        del convolution_47
        del le_36
        del primals_143
        del squeeze_142
        del unsqueeze_786
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf277 = aten.convolution_backward(buf276, add_254, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_254
        del primals_142
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = buf274; del buf274  # reuse
        buf281 = empty((128, ), device='cuda', dtype=torch.float32)
        buf283 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_29.run(le_37, buf271, buf278, convolution_46, unsqueeze_798, squeeze_139, buf280, buf281, buf283, 128, 1568, grid=grid(128), stream=stream0)
        buf282 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_30.run(le_37, buf271, buf278, convolution_46, unsqueeze_798, buf281, squeeze_139, buf280, primals_140, buf282, 200704, grid=grid(200704), stream=stream0)
        del convolution_46
        del le_37
        del primals_140
        del squeeze_139
        del unsqueeze_798
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf284 = aten.convolution_backward(buf282, add_248, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_248
        del primals_139
        buf285 = buf284[0]
        buf286 = buf284[1]
        del buf284
        buf287 = buf281; del buf281  # reuse
        buf288 = empty((128, ), device='cuda', dtype=torch.float32)
        buf290 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_31.run(le_38, buf271, buf285, convolution_45, unsqueeze_810, squeeze_136, buf287, buf288, buf290, 128, 1568, grid=grid(128), stream=stream0)
        buf289 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32.run(le_38, buf271, buf285, convolution_45, unsqueeze_810, buf288, squeeze_136, buf287, primals_137, buf289, 200704, grid=grid(200704), stream=stream0)
        del convolution_45
        del le_38
        del primals_137
        del squeeze_136
        del unsqueeze_810
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf291 = aten.convolution_backward(buf289, getitem_256, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf289
        del getitem_256
        del primals_136
        buf292 = buf291[0]
        buf293 = buf291[1]
        del buf291
        buf294 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_33.run(buf294, le_39, buf292, buf285, buf278, 802816, grid=grid(802816), stream=stream0)
        del buf278
        del buf285
        del le_39
        buf295 = buf260; del buf260  # reuse
        buf296 = empty((512, ), device='cuda', dtype=torch.float32)
        buf297 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_34.run(buf294, convolution_44, unsqueeze_822, squeeze_133, buf295, buf296, buf297, 512, 1568, grid=grid(512), stream=stream0)
        buf298 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_35.run(buf298, convolution_44, unsqueeze_822, buf296, squeeze_133, buf295, primals_134, 802816, grid=grid(802816), stream=stream0)
        del convolution_44
        del primals_134
        del squeeze_133
        del unsqueeze_822
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf299 = aten.convolution_backward(buf298, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_133
        buf300 = buf299[0]
        buf301 = buf299[1]
        del buf299
        buf302 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_38.run(buf302, relu_40, relu_45, buf264, buf300, 1605632, grid=grid(1605632), stream=stream0)
        del relu_40
        del relu_45
        buf303 = buf267; del buf267  # reuse
        buf304 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf310 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf305 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf311 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_39.run(buf302, convolution_43, unsqueeze_834, convolution_42, unsqueeze_846, squeeze_130, squeeze_127, buf303, buf304, buf310, buf305, buf311, 1024, 1568, grid=grid(1024), stream=stream0)
        buf306 = buf300; del buf300  # reuse
        buf312 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_40.run(buf302, convolution_43, unsqueeze_834, buf304, squeeze_130, buf303, primals_131, convolution_42, unsqueeze_846, buf310, squeeze_127, primals_128, buf306, buf312, 1605632, grid=grid(1605632), stream=stream0)
        del buf302
        del buf304
        del buf310
        del convolution_42
        del convolution_43
        del primals_128
        del primals_131
        del squeeze_127
        del squeeze_130
        del unsqueeze_834
        del unsqueeze_846
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf307 = aten.convolution_backward(buf306, relu_35, primals_130, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf306
        del primals_130
        buf308 = buf307[0]
        buf309 = buf307[1]
        del buf307
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf313 = aten.convolution_backward(buf312, cat_7, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf312
        del cat_7
        del primals_127
        buf314 = buf313[0]
        buf315 = buf313[1]
        del buf313
        buf316 = reinterpret_tensor(buf298, (8, 128, 28, 28), (100352, 784, 28, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_41.run(buf314, buf316, 802816, grid=grid(802816), stream=stream0)
        buf317 = buf288; del buf288  # reuse
        buf318 = empty((128, ), device='cuda', dtype=torch.float32)
        buf319 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(le_41, buf314, convolution_41, unsqueeze_858, squeeze_124, buf317, buf318, buf319, 128, 1568, grid=grid(128), stream=stream0)
        buf320 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28.run(le_41, buf314, convolution_41, unsqueeze_858, buf318, squeeze_124, buf317, primals_125, buf320, 200704, grid=grid(200704), stream=stream0)
        del convolution_41
        del le_41
        del primals_125
        del squeeze_124
        del unsqueeze_858
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf321 = aten.convolution_backward(buf320, getitem_238, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del getitem_238
        del primals_124
        buf322 = buf321[0]
        buf323 = buf321[1]
        del buf321
        buf324 = buf318; del buf318  # reuse
        buf325 = empty((128, ), device='cuda', dtype=torch.float32)
        buf326 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_42.run(le_42, buf314, convolution_40, unsqueeze_870, squeeze_121, buf324, buf325, buf326, 128, 1568, grid=grid(128), stream=stream0)
        buf327 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_42, buf314, convolution_40, unsqueeze_870, buf325, squeeze_121, buf324, primals_122, buf327, 200704, grid=grid(200704), stream=stream0)
        del convolution_40
        del le_42
        del primals_122
        del squeeze_121
        del unsqueeze_870
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf328 = aten.convolution_backward(buf327, getitem_231, primals_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del getitem_231
        del primals_121
        buf329 = buf328[0]
        buf330 = buf328[1]
        del buf328
        buf331 = buf325; del buf325  # reuse
        buf332 = empty((128, ), device='cuda', dtype=torch.float32)
        buf333 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(le_43, buf314, convolution_39, unsqueeze_882, squeeze_118, buf331, buf332, buf333, 128, 1568, grid=grid(128), stream=stream0)
        buf334 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(le_43, buf314, convolution_39, unsqueeze_882, buf332, squeeze_118, buf331, primals_119, buf334, 200704, grid=grid(200704), stream=stream0)
        del buf314
        del convolution_39
        del le_43
        del primals_119
        del squeeze_118
        del unsqueeze_882
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf335 = aten.convolution_backward(buf334, getitem_224, primals_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf334
        del getitem_224
        del primals_118
        buf336 = buf335[0]
        buf337 = buf335[1]
        del buf335
        buf338 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_46.run(le_44, buf336, buf329, buf322, buf316, buf338, 3211264, grid=grid(3211264), stream=stream0)
        del buf316
        del buf322
        del buf329
        del le_44
        buf339 = buf296; del buf296  # reuse
        buf340 = empty((512, ), device='cuda', dtype=torch.float32)
        buf341 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_47.run(buf338, convolution_38, unsqueeze_894, squeeze_115, buf339, buf340, buf341, 512, 6272, grid=grid(512), stream=stream0)
        buf342 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_48.run(buf342, convolution_38, unsqueeze_894, buf340, squeeze_115, buf339, primals_116, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_38
        del primals_116
        del squeeze_115
        del unsqueeze_894
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf343 = aten.convolution_backward(buf342, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_115
        buf344 = buf343[0]
        buf345 = buf343[1]
        del buf343
        buf346 = buf340; del buf340  # reuse
        buf347 = empty((512, ), device='cuda', dtype=torch.float32)
        buf349 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(relu_35, buf308, buf344, convolution_37, unsqueeze_906, squeeze_112, buf346, buf347, buf349, 512, 6272, grid=grid(512), stream=stream0)
        buf348 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(relu_35, buf308, buf344, convolution_37, unsqueeze_906, buf347, squeeze_112, buf346, primals_113, buf348, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_37
        del primals_113
        del squeeze_112
        del unsqueeze_906
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf350 = aten.convolution_backward(buf348, cat_6, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf348
        del cat_6
        del primals_112
        buf351 = buf350[0]
        buf352 = buf350[1]
        del buf350
        buf353 = empty((64, ), device='cuda', dtype=torch.float32)
        buf354 = empty((64, ), device='cuda', dtype=torch.float32)
        buf355 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_51.run(le_46, buf351, convolution_36, unsqueeze_918, squeeze_109, buf353, buf354, buf355, 64, 6272, grid=grid(64), stream=stream0)
        buf356 = reinterpret_tensor(buf96, (8, 64, 28, 28), (50176, 784, 28, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52.run(le_46, buf351, convolution_36, unsqueeze_918, buf354, squeeze_109, buf353, primals_110, buf356, 401408, grid=grid(401408), stream=stream0)
        del convolution_36
        del le_46
        del primals_110
        del squeeze_109
        del unsqueeze_918
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf357 = aten.convolution_backward(buf356, add_195, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_195
        del primals_109
        buf358 = buf357[0]
        buf359 = buf357[1]
        del buf357
        buf360 = buf354; del buf354  # reuse
        buf361 = empty((64, ), device='cuda', dtype=torch.float32)
        buf363 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_53.run(le_47, buf351, buf358, convolution_35, unsqueeze_930, squeeze_106, buf360, buf361, buf363, 64, 6272, grid=grid(64), stream=stream0)
        buf362 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_54.run(le_47, buf351, buf358, convolution_35, unsqueeze_930, buf361, squeeze_106, buf360, primals_107, buf362, 401408, grid=grid(401408), stream=stream0)
        del convolution_35
        del le_47
        del primals_107
        del squeeze_106
        del unsqueeze_930
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf364 = aten.convolution_backward(buf362, add_189, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_189
        del primals_106
        buf365 = buf364[0]
        buf366 = buf364[1]
        del buf364
        buf367 = buf361; del buf361  # reuse
        buf368 = empty((64, ), device='cuda', dtype=torch.float32)
        buf370 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(le_48, buf351, buf365, convolution_34, unsqueeze_942, squeeze_103, buf367, buf368, buf370, 64, 6272, grid=grid(64), stream=stream0)
        buf369 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_56.run(le_48, buf351, buf365, convolution_34, unsqueeze_942, buf368, squeeze_103, buf367, primals_104, buf369, 401408, grid=grid(401408), stream=stream0)
        del convolution_34
        del le_48
        del primals_104
        del squeeze_103
        del unsqueeze_942
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf371 = aten.convolution_backward(buf369, getitem_194, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf369
        del getitem_194
        del primals_103
        buf372 = buf371[0]
        buf373 = buf371[1]
        del buf371
        buf374 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_57.run(buf374, le_49, buf372, buf365, buf358, 1605632, grid=grid(1605632), stream=stream0)
        del buf358
        del buf365
        del le_49
        buf375 = buf106; del buf106  # reuse
        buf376 = empty((256, ), device='cuda', dtype=torch.float32)
        buf377 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf374, convolution_33, unsqueeze_954, squeeze_100, buf375, buf376, buf377, 256, 6272, grid=grid(256), stream=stream0)
        buf378 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf378, convolution_33, unsqueeze_954, buf376, squeeze_100, buf375, primals_101, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_33
        del primals_101
        del squeeze_100
        del unsqueeze_954
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf379 = aten.convolution_backward(buf378, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf378
        del primals_100
        buf380 = buf379[0]
        buf381 = buf379[1]
        del buf379
        buf382 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf382, relu_30, relu_35, buf344, buf380, 3211264, grid=grid(3211264), stream=stream0)
        del buf344
        del relu_30
        del relu_35
        buf383 = buf347; del buf347  # reuse
        buf384 = empty((512, ), device='cuda', dtype=torch.float32)
        buf385 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_47.run(buf382, convolution_32, unsqueeze_966, squeeze_97, buf383, buf384, buf385, 512, 6272, grid=grid(512), stream=stream0)
        buf386 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_61.run(buf382, convolution_32, unsqueeze_966, buf384, squeeze_97, buf383, primals_98, buf386, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_32
        del primals_98
        del squeeze_97
        del unsqueeze_966
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf387 = aten.convolution_backward(buf386, cat_5, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_5
        del primals_97
        buf388 = buf387[0]
        buf389 = buf387[1]
        del buf387
        buf390 = buf368; del buf368  # reuse
        buf391 = empty((64, ), device='cuda', dtype=torch.float32)
        buf392 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_51.run(le_51, buf388, convolution_31, unsqueeze_978, squeeze_94, buf390, buf391, buf392, 64, 6272, grid=grid(64), stream=stream0)
        buf393 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52.run(le_51, buf388, convolution_31, unsqueeze_978, buf391, squeeze_94, buf390, primals_95, buf393, 401408, grid=grid(401408), stream=stream0)
        del convolution_31
        del le_51
        del primals_95
        del squeeze_94
        del unsqueeze_978
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf394 = aten.convolution_backward(buf393, add_167, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_167
        del primals_94
        buf395 = buf394[0]
        buf396 = buf394[1]
        del buf394
        buf397 = buf391; del buf391  # reuse
        buf398 = empty((64, ), device='cuda', dtype=torch.float32)
        buf400 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_53.run(le_52, buf388, buf395, convolution_30, unsqueeze_990, squeeze_91, buf397, buf398, buf400, 64, 6272, grid=grid(64), stream=stream0)
        buf399 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_54.run(le_52, buf388, buf395, convolution_30, unsqueeze_990, buf398, squeeze_91, buf397, primals_92, buf399, 401408, grid=grid(401408), stream=stream0)
        del convolution_30
        del le_52
        del primals_92
        del squeeze_91
        del unsqueeze_990
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf401 = aten.convolution_backward(buf399, add_161, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_161
        del primals_91
        buf402 = buf401[0]
        buf403 = buf401[1]
        del buf401
        buf404 = buf398; del buf398  # reuse
        buf405 = empty((64, ), device='cuda', dtype=torch.float32)
        buf407 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(le_53, buf388, buf402, convolution_29, unsqueeze_1002, squeeze_88, buf404, buf405, buf407, 64, 6272, grid=grid(64), stream=stream0)
        buf406 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_56.run(le_53, buf388, buf402, convolution_29, unsqueeze_1002, buf405, squeeze_88, buf404, primals_89, buf406, 401408, grid=grid(401408), stream=stream0)
        del convolution_29
        del le_53
        del primals_89
        del squeeze_88
        del unsqueeze_1002
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf408 = aten.convolution_backward(buf406, getitem_164, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf406
        del getitem_164
        del primals_88
        buf409 = buf408[0]
        buf410 = buf408[1]
        del buf408
        buf411 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_57.run(buf411, le_54, buf409, buf402, buf395, 1605632, grid=grid(1605632), stream=stream0)
        del buf395
        del buf402
        del le_54
        buf412 = buf376; del buf376  # reuse
        buf413 = empty((256, ), device='cuda', dtype=torch.float32)
        buf414 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf411, convolution_28, unsqueeze_1014, squeeze_85, buf412, buf413, buf414, 256, 6272, grid=grid(256), stream=stream0)
        buf415 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf415, convolution_28, unsqueeze_1014, buf413, squeeze_85, buf412, primals_86, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_28
        del primals_86
        del squeeze_85
        del unsqueeze_1014
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf416 = aten.convolution_backward(buf415, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf415
        del primals_85
        buf417 = buf416[0]
        buf418 = buf416[1]
        del buf416
        buf419 = buf384; del buf384  # reuse
        buf420 = empty((512, ), device='cuda', dtype=torch.float32)
        buf422 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(relu_25, buf382, buf417, convolution_27, unsqueeze_1026, squeeze_82, buf419, buf420, buf422, 512, 6272, grid=grid(512), stream=stream0)
        buf421 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(relu_25, buf382, buf417, convolution_27, unsqueeze_1026, buf420, squeeze_82, buf419, primals_83, buf421, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_27
        del primals_83
        del squeeze_82
        del unsqueeze_1026
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf423 = aten.convolution_backward(buf421, cat_4, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf421
        del cat_4
        del primals_82
        buf424 = buf423[0]
        buf425 = buf423[1]
        del buf423
        buf426 = buf405; del buf405  # reuse
        buf427 = empty((64, ), device='cuda', dtype=torch.float32)
        buf428 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_51.run(le_56, buf424, convolution_26, unsqueeze_1038, squeeze_79, buf426, buf427, buf428, 64, 6272, grid=grid(64), stream=stream0)
        buf429 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52.run(le_56, buf424, convolution_26, unsqueeze_1038, buf427, squeeze_79, buf426, primals_80, buf429, 401408, grid=grid(401408), stream=stream0)
        del convolution_26
        del le_56
        del primals_80
        del squeeze_79
        del unsqueeze_1038
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf430 = aten.convolution_backward(buf429, add_139, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_139
        del primals_79
        buf431 = buf430[0]
        buf432 = buf430[1]
        del buf430
        buf433 = buf427; del buf427  # reuse
        buf434 = empty((64, ), device='cuda', dtype=torch.float32)
        buf436 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_53.run(le_57, buf424, buf431, convolution_25, unsqueeze_1050, squeeze_76, buf433, buf434, buf436, 64, 6272, grid=grid(64), stream=stream0)
        buf435 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_54.run(le_57, buf424, buf431, convolution_25, unsqueeze_1050, buf434, squeeze_76, buf433, primals_77, buf435, 401408, grid=grid(401408), stream=stream0)
        del convolution_25
        del le_57
        del primals_77
        del squeeze_76
        del unsqueeze_1050
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf437 = aten.convolution_backward(buf435, add_133, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_133
        del primals_76
        buf438 = buf437[0]
        buf439 = buf437[1]
        del buf437
        buf440 = buf434; del buf434  # reuse
        buf441 = empty((64, ), device='cuda', dtype=torch.float32)
        buf443 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_55.run(le_58, buf424, buf438, convolution_24, unsqueeze_1062, squeeze_73, buf440, buf441, buf443, 64, 6272, grid=grid(64), stream=stream0)
        buf442 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_56.run(le_58, buf424, buf438, convolution_24, unsqueeze_1062, buf441, squeeze_73, buf440, primals_74, buf442, 401408, grid=grid(401408), stream=stream0)
        del convolution_24
        del le_58
        del primals_74
        del squeeze_73
        del unsqueeze_1062
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf444 = aten.convolution_backward(buf442, getitem_134, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf442
        del getitem_134
        del primals_73
        buf445 = buf444[0]
        buf446 = buf444[1]
        del buf444
        buf447 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_57.run(buf447, le_59, buf445, buf438, buf431, 1605632, grid=grid(1605632), stream=stream0)
        del buf431
        del buf438
        del le_59
        buf448 = buf413; del buf413  # reuse
        buf449 = empty((256, ), device='cuda', dtype=torch.float32)
        buf450 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf447, convolution_23, unsqueeze_1074, squeeze_70, buf448, buf449, buf450, 256, 6272, grid=grid(256), stream=stream0)
        buf451 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf451, convolution_23, unsqueeze_1074, buf449, squeeze_70, buf448, primals_71, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_23
        del primals_71
        del squeeze_70
        del unsqueeze_1074
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf452 = aten.convolution_backward(buf451, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_70
        buf453 = buf452[0]
        buf454 = buf452[1]
        del buf452
        buf455 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf455, relu_20, relu_25, buf417, buf453, 3211264, grid=grid(3211264), stream=stream0)
        del relu_20
        del relu_25
        buf456 = buf420; del buf420  # reuse
        buf457 = empty((512, ), device='cuda', dtype=torch.float32)
        buf463 = empty((512, ), device='cuda', dtype=torch.float32)
        buf458 = empty((512, ), device='cuda', dtype=torch.float32)
        buf464 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_62.run(buf455, convolution_22, unsqueeze_1086, convolution_21, unsqueeze_1098, squeeze_67, squeeze_64, buf456, buf457, buf463, buf458, buf464, 512, 6272, grid=grid(512), stream=stream0)
        buf459 = buf453; del buf453  # reuse
        buf465 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_63.run(buf455, convolution_22, unsqueeze_1086, buf457, squeeze_67, buf456, primals_68, convolution_21, unsqueeze_1098, buf463, squeeze_64, primals_65, buf459, buf465, 3211264, grid=grid(3211264), stream=stream0)
        del buf455
        del convolution_21
        del convolution_22
        del primals_65
        del primals_68
        del squeeze_64
        del squeeze_67
        del unsqueeze_1086
        del unsqueeze_1098
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf460 = aten.convolution_backward(buf459, relu_15, primals_67, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf459
        del primals_67
        buf461 = buf460[0]
        buf462 = buf460[1]
        del buf460
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf466 = aten.convolution_backward(buf465, cat_3, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf465
        del cat_3
        del primals_64
        buf467 = buf466[0]
        buf468 = buf466[1]
        del buf466
        buf469 = reinterpret_tensor(buf451, (8, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_64.run(buf467, buf469, 1605632, grid=grid(1605632), stream=stream0)
        buf470 = buf441; del buf441  # reuse
        buf471 = empty((64, ), device='cuda', dtype=torch.float32)
        buf472 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_51.run(le_61, buf467, convolution_20, unsqueeze_1110, squeeze_61, buf470, buf471, buf472, 64, 6272, grid=grid(64), stream=stream0)
        buf473 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52.run(le_61, buf467, convolution_20, unsqueeze_1110, buf471, squeeze_61, buf470, primals_62, buf473, 401408, grid=grid(401408), stream=stream0)
        del convolution_20
        del le_61
        del primals_62
        del squeeze_61
        del unsqueeze_1110
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf474 = aten.convolution_backward(buf473, getitem_116, primals_61, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del getitem_116
        del primals_61
        buf475 = buf474[0]
        buf476 = buf474[1]
        del buf474
        buf477 = buf471; del buf471  # reuse
        buf478 = empty((64, ), device='cuda', dtype=torch.float32)
        buf479 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(le_62, buf467, convolution_19, unsqueeze_1122, squeeze_58, buf477, buf478, buf479, 64, 6272, grid=grid(64), stream=stream0)
        buf480 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66.run(le_62, buf467, convolution_19, unsqueeze_1122, buf478, squeeze_58, buf477, primals_59, buf480, 401408, grid=grid(401408), stream=stream0)
        del convolution_19
        del le_62
        del primals_59
        del squeeze_58
        del unsqueeze_1122
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf481 = aten.convolution_backward(buf480, getitem_109, primals_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del getitem_109
        del primals_58
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        buf484 = buf478; del buf478  # reuse
        buf485 = empty((64, ), device='cuda', dtype=torch.float32)
        buf486 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_67.run(le_63, buf467, convolution_18, unsqueeze_1134, squeeze_55, buf484, buf485, buf486, 64, 6272, grid=grid(64), stream=stream0)
        buf487 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68.run(le_63, buf467, convolution_18, unsqueeze_1134, buf485, squeeze_55, buf484, primals_56, buf487, 401408, grid=grid(401408), stream=stream0)
        del buf467
        del convolution_18
        del le_63
        del primals_56
        del squeeze_55
        del unsqueeze_1134
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf488 = aten.convolution_backward(buf487, getitem_102, primals_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf487
        del getitem_102
        del primals_55
        buf489 = buf488[0]
        buf490 = buf488[1]
        del buf488
        buf491 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_69.run(le_64, buf489, buf482, buf475, buf469, buf491, 6422528, grid=grid(6422528), stream=stream0)
        del buf469
        del buf475
        del buf482
        del buf489
        del le_64
        buf492 = buf449; del buf449  # reuse
        buf493 = empty((256, ), device='cuda', dtype=torch.float32)
        buf494 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_70.run(buf491, convolution_17, unsqueeze_1146, squeeze_52, buf492, buf493, buf494, 256, 25088, grid=grid(256), stream=stream0)
        buf495 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_71.run(buf495, convolution_17, unsqueeze_1146, buf493, squeeze_52, buf492, primals_53, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_17
        del primals_53
        del squeeze_52
        del unsqueeze_1146
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf496 = aten.convolution_backward(buf495, relu_15, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_52
        buf497 = buf496[0]
        buf498 = buf496[1]
        del buf496
        buf499 = buf493; del buf493  # reuse
        buf500 = empty((256, ), device='cuda', dtype=torch.float32)
        buf502 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_72.run(relu_15, buf461, buf497, convolution_16, unsqueeze_1158, squeeze_49, buf499, buf500, buf502, 256, 25088, grid=grid(256), stream=stream0)
        buf501 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_73.run(relu_15, buf461, buf497, convolution_16, unsqueeze_1158, buf500, squeeze_49, buf499, primals_50, buf501, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_16
        del primals_50
        del squeeze_49
        del unsqueeze_1158
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf503 = aten.convolution_backward(buf501, cat_2, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf501
        del cat_2
        del primals_49
        buf504 = buf503[0]
        buf505 = buf503[1]
        del buf503
        buf506 = reinterpret_tensor(buf332, (32, 4), (1, 32), 0); del buf332  # reuse
        buf508 = empty_strided((32, 4), (1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(le_66, buf504, convolution_15, unsqueeze_1170, buf506, buf508, 128, 6272, grid=grid(128), stream=stream0)
        buf507 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf506, buf507, 32, 4, grid=grid(32), stream=stream0)
        buf509 = empty((32, ), device='cuda', dtype=torch.float32)
        buf510 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf508, squeeze_46, buf509, buf510, 32, 4, grid=grid(32), stream=stream0)
        buf511 = reinterpret_tensor(buf336, (8, 32, 56, 56), (100352, 3136, 56, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77.run(le_66, buf504, convolution_15, unsqueeze_1170, buf509, squeeze_46, buf507, primals_47, buf511, 802816, grid=grid(802816), stream=stream0)
        del convolution_15
        del le_66
        del primals_47
        del squeeze_46
        del unsqueeze_1170
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf512 = aten.convolution_backward(buf511, add_80, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_80
        del primals_46
        buf513 = buf512[0]
        buf514 = buf512[1]
        del buf512
        buf515 = buf508; del buf508  # reuse
        buf517 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_78.run(le_67, buf504, buf513, convolution_14, unsqueeze_1182, buf515, buf517, 128, 6272, grid=grid(128), stream=stream0)
        buf516 = buf509; del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf515, buf516, 32, 4, grid=grid(32), stream=stream0)
        buf518 = empty((32, ), device='cuda', dtype=torch.float32)
        buf520 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf517, squeeze_43, buf518, buf520, 32, 4, grid=grid(32), stream=stream0)
        buf519 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_79.run(le_67, buf504, buf513, convolution_14, unsqueeze_1182, buf518, squeeze_43, buf516, primals_44, buf519, 802816, grid=grid(802816), stream=stream0)
        del convolution_14
        del le_67
        del primals_44
        del squeeze_43
        del unsqueeze_1182
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf521 = aten.convolution_backward(buf519, add_74, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_74
        del primals_43
        buf522 = buf521[0]
        buf523 = buf521[1]
        del buf521
        buf524 = buf517; del buf517  # reuse
        buf526 = buf515; del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_80.run(le_68, buf504, buf522, convolution_13, unsqueeze_1194, buf524, buf526, 128, 6272, grid=grid(128), stream=stream0)
        buf525 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf524, buf525, 32, 4, grid=grid(32), stream=stream0)
        buf527 = empty((32, ), device='cuda', dtype=torch.float32)
        buf529 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf526, squeeze_40, buf527, buf529, 32, 4, grid=grid(32), stream=stream0)
        buf528 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_81.run(le_68, buf504, buf522, convolution_13, unsqueeze_1194, buf527, squeeze_40, buf525, primals_41, buf528, 802816, grid=grid(802816), stream=stream0)
        del convolution_13
        del le_68
        del primals_41
        del squeeze_40
        del unsqueeze_1194
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf530 = aten.convolution_backward(buf528, getitem_72, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf528
        del getitem_72
        del primals_40
        buf531 = buf530[0]
        buf532 = buf530[1]
        del buf530
        buf533 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_82.run(buf533, le_69, buf531, buf522, buf513, 3211264, grid=grid(3211264), stream=stream0)
        del buf513
        del buf522
        del le_69
        buf534 = reinterpret_tensor(buf463, (128, 4), (1, 128), 0); del buf463  # reuse
        buf536 = reinterpret_tensor(buf457, (128, 4), (1, 128), 0); del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_83.run(buf533, convolution_12, unsqueeze_1206, buf534, buf536, 512, 6272, grid=grid(512), stream=stream0)
        buf535 = reinterpret_tensor(buf526, (128, ), (1, ), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf534, buf535, 128, 4, grid=grid(128), stream=stream0)
        buf537 = reinterpret_tensor(buf524, (128, ), (1, ), 0); del buf524  # reuse
        buf538 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf536, squeeze_37, buf537, buf538, 128, 4, grid=grid(128), stream=stream0)
        buf539 = buf533; del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_86.run(buf539, convolution_12, unsqueeze_1206, buf537, squeeze_37, buf535, primals_38, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_12
        del primals_38
        del squeeze_37
        del unsqueeze_1206
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf540 = aten.convolution_backward(buf539, relu_10, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf539
        del primals_37
        buf541 = buf540[0]
        buf542 = buf540[1]
        del buf540
        buf543 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_87.run(buf543, relu_10, relu_15, buf497, buf541, 6422528, grid=grid(6422528), stream=stream0)
        del relu_10
        del relu_15
        buf544 = buf500; del buf500  # reuse
        buf545 = empty((256, ), device='cuda', dtype=torch.float32)
        buf546 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_70.run(buf543, convolution_11, unsqueeze_1218, squeeze_34, buf544, buf545, buf546, 256, 25088, grid=grid(256), stream=stream0)
        buf547 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_88.run(buf543, convolution_11, unsqueeze_1218, buf545, squeeze_34, buf544, primals_35, buf547, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_11
        del primals_35
        del squeeze_34
        del unsqueeze_1218
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf548 = aten.convolution_backward(buf547, cat_1, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_34
        buf549 = buf548[0]
        buf550 = buf548[1]
        del buf548
        buf551 = reinterpret_tensor(buf537, (32, 4), (1, 32), 0); del buf537  # reuse
        buf553 = empty_strided((32, 4), (1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(le_71, buf549, convolution_10, unsqueeze_1230, buf551, buf553, 128, 6272, grid=grid(128), stream=stream0)
        buf552 = buf527; del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf551, buf552, 32, 4, grid=grid(32), stream=stream0)
        buf554 = empty((32, ), device='cuda', dtype=torch.float32)
        buf555 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf553, squeeze_31, buf554, buf555, 32, 4, grid=grid(32), stream=stream0)
        buf556 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77.run(le_71, buf549, convolution_10, unsqueeze_1230, buf554, squeeze_31, buf552, primals_32, buf556, 802816, grid=grid(802816), stream=stream0)
        del convolution_10
        del le_71
        del primals_32
        del squeeze_31
        del unsqueeze_1230
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf557 = aten.convolution_backward(buf556, add_52, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_52
        del primals_31
        buf558 = buf557[0]
        buf559 = buf557[1]
        del buf557
        buf560 = buf553; del buf553  # reuse
        buf562 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_78.run(le_72, buf549, buf558, convolution_9, unsqueeze_1242, buf560, buf562, 128, 6272, grid=grid(128), stream=stream0)
        buf561 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf560, buf561, 32, 4, grid=grid(32), stream=stream0)
        buf563 = empty((32, ), device='cuda', dtype=torch.float32)
        buf565 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf562, squeeze_28, buf563, buf565, 32, 4, grid=grid(32), stream=stream0)
        buf564 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_79.run(le_72, buf549, buf558, convolution_9, unsqueeze_1242, buf563, squeeze_28, buf561, primals_29, buf564, 802816, grid=grid(802816), stream=stream0)
        del convolution_9
        del le_72
        del primals_29
        del squeeze_28
        del unsqueeze_1242
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf566 = aten.convolution_backward(buf564, add_46, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del add_46
        del primals_28
        buf567 = buf566[0]
        buf568 = buf566[1]
        del buf566
        buf569 = buf562; del buf562  # reuse
        buf571 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_80.run(le_73, buf549, buf567, convolution_8, unsqueeze_1254, buf569, buf571, 128, 6272, grid=grid(128), stream=stream0)
        buf570 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf569, buf570, 32, 4, grid=grid(32), stream=stream0)
        buf572 = empty((32, ), device='cuda', dtype=torch.float32)
        buf574 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf571, squeeze_25, buf572, buf574, 32, 4, grid=grid(32), stream=stream0)
        buf573 = buf564; del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_81.run(le_73, buf549, buf567, convolution_8, unsqueeze_1254, buf572, squeeze_25, buf570, primals_26, buf573, 802816, grid=grid(802816), stream=stream0)
        del convolution_8
        del le_73
        del primals_26
        del squeeze_25
        del unsqueeze_1254
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf575 = aten.convolution_backward(buf573, getitem_42, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf573
        del getitem_42
        del primals_25
        buf576 = buf575[0]
        buf577 = buf575[1]
        del buf575
        buf578 = buf549; del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_82.run(buf578, le_74, buf576, buf567, buf558, 3211264, grid=grid(3211264), stream=stream0)
        del buf558
        del le_74
        buf579 = buf536; del buf536  # reuse
        buf581 = buf534; del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_83.run(buf578, convolution_7, unsqueeze_1266, buf579, buf581, 512, 6272, grid=grid(512), stream=stream0)
        buf580 = reinterpret_tensor(buf571, (128, ), (1, ), 0); del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf579, buf580, 128, 4, grid=grid(128), stream=stream0)
        buf582 = reinterpret_tensor(buf569, (128, ), (1, ), 0); del buf569  # reuse
        buf583 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf581, squeeze_22, buf582, buf583, 128, 4, grid=grid(128), stream=stream0)
        buf584 = buf578; del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_86.run(buf584, convolution_7, unsqueeze_1266, buf582, squeeze_22, buf580, primals_23, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_7
        del primals_23
        del squeeze_22
        del unsqueeze_1266
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf585 = aten.convolution_backward(buf584, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf584
        del primals_22
        buf586 = buf585[0]
        buf587 = buf585[1]
        del buf585
        buf588 = buf545; del buf545  # reuse
        buf589 = empty((256, ), device='cuda', dtype=torch.float32)
        buf595 = empty((256, ), device='cuda', dtype=torch.float32)
        buf591 = empty((256, ), device='cuda', dtype=torch.float32)
        buf597 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_89.run(relu_5, buf543, buf586, convolution_6, unsqueeze_1278, convolution_5, unsqueeze_1290, squeeze_19, squeeze_16, buf588, buf589, buf595, buf591, buf597, 256, 25088, grid=grid(256), stream=stream0)
        buf590 = buf547; del buf547  # reuse
        buf596 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_90.run(relu_5, buf543, buf586, convolution_6, unsqueeze_1278, buf589, squeeze_19, buf588, primals_20, convolution_5, unsqueeze_1290, buf595, squeeze_16, primals_17, buf590, buf596, 6422528, grid=grid(6422528), stream=stream0)
        del buf543
        del buf586
        del buf589
        del buf595
        del convolution_5
        del convolution_6
        del primals_17
        del primals_20
        del relu_5
        del squeeze_16
        del squeeze_19
        del unsqueeze_1278
        del unsqueeze_1290
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf592 = aten.convolution_backward(buf590, getitem_2, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf590
        del primals_19
        buf593 = buf592[0]
        buf594 = buf592[1]
        del buf592
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf598 = aten.convolution_backward(buf596, cat, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat
        del primals_16
        buf599 = buf598[0]
        buf600 = buf598[1]
        del buf598
        buf601 = buf576; del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_91.run(buf599, buf601, 802816, grid=grid(802816), stream=stream0)
        buf602 = reinterpret_tensor(buf582, (32, 4), (1, 32), 0); del buf582  # reuse
        buf604 = empty_strided((32, 4), (1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(le_76, buf599, convolution_4, unsqueeze_1302, buf602, buf604, 128, 6272, grid=grid(128), stream=stream0)
        buf603 = buf572; del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf602, buf603, 32, 4, grid=grid(32), stream=stream0)
        buf605 = empty((32, ), device='cuda', dtype=torch.float32)
        buf606 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf604, squeeze_13, buf605, buf606, 32, 4, grid=grid(32), stream=stream0)
        buf607 = buf567; del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77.run(le_76, buf599, convolution_4, unsqueeze_1302, buf605, squeeze_13, buf603, primals_14, buf607, 802816, grid=grid(802816), stream=stream0)
        del convolution_4
        del le_76
        del primals_14
        del squeeze_13
        del unsqueeze_1302
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf608 = aten.convolution_backward(buf607, getitem_24, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del getitem_24
        del primals_13
        buf609 = buf608[0]
        buf610 = buf608[1]
        del buf608
        buf611 = buf604; del buf604  # reuse
        buf613 = buf602; del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_92.run(le_77, buf599, convolution_3, unsqueeze_1314, buf611, buf613, 128, 6272, grid=grid(128), stream=stream0)
        buf612 = buf605; del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf611, buf612, 32, 4, grid=grid(32), stream=stream0)
        buf614 = empty((32, ), device='cuda', dtype=torch.float32)
        buf615 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf613, squeeze_10, buf614, buf615, 32, 4, grid=grid(32), stream=stream0)
        buf616 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93.run(le_77, buf599, convolution_3, unsqueeze_1314, buf614, squeeze_10, buf612, primals_11, buf616, 802816, grid=grid(802816), stream=stream0)
        del convolution_3
        del le_77
        del primals_11
        del squeeze_10
        del unsqueeze_1314
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf617 = aten.convolution_backward(buf616, getitem_17, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del getitem_17
        del primals_10
        buf618 = buf617[0]
        buf619 = buf617[1]
        del buf617
        buf620 = buf613; del buf613  # reuse
        buf622 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_94.run(le_78, buf599, convolution_2, unsqueeze_1326, buf620, buf622, 128, 6272, grid=grid(128), stream=stream0)
        buf621 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf620, buf621, 32, 4, grid=grid(32), stream=stream0)
        buf623 = empty((32, ), device='cuda', dtype=torch.float32)
        buf624 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_76.run(buf622, squeeze_7, buf623, buf624, 32, 4, grid=grid(32), stream=stream0)
        buf625 = buf616; del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_95.run(le_78, buf599, convolution_2, unsqueeze_1326, buf623, squeeze_7, buf621, primals_8, buf625, 802816, grid=grid(802816), stream=stream0)
        del buf623
        del convolution_2
        del le_78
        del primals_8
        del squeeze_7
        del unsqueeze_1326
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf626 = aten.convolution_backward(buf625, getitem_10, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf625
        del getitem_10
        del primals_7
        buf627 = buf626[0]
        buf628 = buf626[1]
        del buf626
        buf629 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_96.run(le_79, buf627, buf618, buf609, buf601, buf629, 3211264, grid=grid(3211264), stream=stream0)
        del buf601
        del buf609
        del buf618
        del buf627
        del le_79
        buf630 = buf581; del buf581  # reuse
        buf632 = buf579; del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_83.run(buf629, convolution_1, unsqueeze_1338, buf630, buf632, 512, 6272, grid=grid(512), stream=stream0)
        buf631 = reinterpret_tensor(buf622, (128, ), (1, ), 0); del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf630, buf631, 128, 4, grid=grid(128), stream=stream0)
        del buf630
        buf633 = reinterpret_tensor(buf620, (128, ), (1, ), 0); del buf620  # reuse
        buf634 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf632, squeeze_4, buf633, buf634, 128, 4, grid=grid(128), stream=stream0)
        del buf632
        buf635 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_86.run(buf635, convolution_1, unsqueeze_1338, buf633, squeeze_4, buf631, primals_5, 3211264, grid=grid(3211264), stream=stream0)
        del buf633
        del convolution_1
        del primals_5
        del squeeze_4
        del unsqueeze_1338
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf636 = aten.convolution_backward(buf635, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf635
        del getitem_2
        del primals_4
        buf637 = buf636[0]
        buf638 = buf636[1]
        del buf636
        buf639 = buf593; del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_97.run(buf639, buf637, 1605632, grid=grid(1605632), stream=stream0)
        del buf637
        buf640 = reinterpret_tensor(buf596, (8, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        triton_poi_fused_add_max_pool2d_with_indices_backward_98.run(getitem_3, buf639, buf640, 6422528, grid=grid(6422528), stream=stream0)
        del buf639
        del getitem_3
        buf641 = empty((64, 13), device='cuda', dtype=torch.float32)
        buf643 = empty((64, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_99.run(relu, buf640, convolution, unsqueeze_1350, buf641, buf643, 832, 7720, grid=grid(832), stream=stream0)
        buf642 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_100.run(buf641, buf642, 64, 13, grid=grid(64), stream=stream0)
        del buf641
        buf644 = empty((64, ), device='cuda', dtype=torch.float32)
        buf645 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_101.run(buf643, squeeze_1, buf644, buf645, 64, 13, grid=grid(64), stream=stream0)
        del buf643
        buf646 = buf640; del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_102.run(buf646, relu, convolution, unsqueeze_1350, buf644, squeeze_1, buf642, primals_2, 6422528, grid=grid(6422528), stream=stream0)
        del buf644
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_1350
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf647 = aten.convolution_backward(buf646, primals_513, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf646
        del primals_1
        del primals_513
        buf648 = buf647[1]
        return (buf648, buf645, buf642, buf638, buf634, buf631, buf628, buf624, buf621, buf619, buf615, buf612, buf610, buf606, buf603, buf600, buf597, buf588, buf594, buf591, buf588, buf587, buf583, buf580, buf577, buf574, buf570, buf568, buf565, buf561, buf559, buf555, buf552, buf550, buf546, buf544, buf542, buf538, buf535, buf532, buf529, buf525, buf523, buf520, buf516, buf514, buf510, buf507, buf505, buf502, buf499, buf498, buf494, buf492, buf490, buf486, buf484, buf483, buf479, buf477, buf476, buf472, buf470, buf468, buf464, buf456, buf462, buf458, buf456, buf454, buf450, buf448, buf446, buf443, buf440, buf439, buf436, buf433, buf432, buf428, buf426, buf425, buf422, buf419, buf418, buf414, buf412, buf410, buf407, buf404, buf403, buf400, buf397, buf396, buf392, buf390, buf389, buf385, buf383, buf381, buf377, buf375, buf373, buf370, buf367, buf366, buf363, buf360, buf359, buf355, buf353, buf352, buf349, buf346, buf345, buf341, buf339, buf337, buf333, buf331, buf330, buf326, buf324, buf323, buf319, buf317, buf315, buf311, buf303, buf309, buf305, buf303, buf301, buf297, buf295, buf293, buf290, buf287, buf286, buf283, buf280, buf279, buf275, buf273, buf272, buf269, buf266, buf265, buf261, buf259, buf257, buf254, buf251, buf250, buf247, buf244, buf243, buf239, buf237, buf236, buf232, buf230, buf228, buf224, buf222, buf220, buf217, buf214, buf213, buf210, buf207, buf206, buf202, buf200, buf199, buf196, buf193, buf192, buf188, buf186, buf184, buf181, buf178, buf177, buf174, buf171, buf170, buf166, buf164, buf163, buf159, buf157, buf155, buf151, buf149, buf147, buf144, buf141, buf140, buf137, buf134, buf133, buf129, buf127, buf126, buf123, buf120, buf119, buf115, buf113, buf111, buf107, buf105, buf104, buf100, buf98, buf97, buf93, buf91, buf89, buf85, buf77, buf83, buf79, buf77, buf75, buf71, buf69, buf67, buf64, buf61, buf60, buf57, buf54, buf53, buf49, buf47, buf46, buf42, buf39, buf38, buf34, buf32, buf30, buf27, buf24, buf23, buf20, buf17, buf16, buf12, buf10, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.int64)
    convolution_1 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_10 = rand_strided((8, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_24 = rand_strided((8, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((8, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_42 = rand_strided((8, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_46 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_52 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((8, 32, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_74 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_80 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_102 = rand_strided((8, 64, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_109 = rand_strided((8, 64, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 64, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((8, 64, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((8, 64, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_133 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_139 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_164 = rand_strided((8, 64, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_161 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_167 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_194 = rand_strided((8, 64, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_189 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_195 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_224 = rand_strided((8, 128, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_231 = rand_strided((8, 128, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_238 = rand_strided((8, 128, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_245 = rand_strided((8, 128, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_256 = rand_strided((8, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_248 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_254 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_286 = rand_strided((8, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_276 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_282 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_316 = rand_strided((8, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_304 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_310 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_10 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_346 = rand_strided((8, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_332 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_338 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_60 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_376 = rand_strided((8, 128, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_360 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_366 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_205 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_406 = rand_strided((8, 256, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_413 = rand_strided((8, 256, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_420 = rand_strided((8, 256, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_427 = rand_strided((8, 256, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_13 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_438 = rand_strided((8, 256, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_419 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_425 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_14 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_468 = rand_strided((8, 256, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_447 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_453 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_15 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_253 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_342 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_354 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_366 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_378 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_414 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_7 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_426 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_438 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((8, 1024, 7, 7), (50176, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_450 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_12 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_498 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_13 = rand_strided((8, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_510 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_14 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_522 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_16 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_546 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_18 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_570 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_19 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_582 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_21 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_618 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_23 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_630 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_24 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_642 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_26 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_27 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_28 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_690 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_29 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_702 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_31 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_726 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_32 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_738 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_33 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_750 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_34 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_762 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_36 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_786 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_37 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_798 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_38 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_810 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_39 = rand_strided((8, 512, 14, 14), (100352, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_822 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_41 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_858 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_42 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_870 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_43 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_882 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_44 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_894 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_46 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_918 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_47 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_930 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_48 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_942 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_49 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_954 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_966 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_51 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_978 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_52 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_990 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_53 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1002 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_54 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1014 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1026 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_56 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1038 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_57 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1050 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_58 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1062 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_59 = rand_strided((8, 256, 28, 28), (200704, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1074 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1086 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1098 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_61 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1110 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_62 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1122 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_63 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1134 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_64 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1146 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1158 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_66 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1170 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_67 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1182 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_68 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1194 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_69 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1206 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1218 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_71 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1230 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_72 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1242 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_73 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1254 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_74 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1266 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1278 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1290 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_76 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1302 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_77 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1314 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_78 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1326 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_79 = rand_strided((8, 128, 56, 56), (401408, 3136, 56, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_1338 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1350 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_513, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, getitem_10, convolution_2, squeeze_7, getitem_17, convolution_3, squeeze_10, getitem_24, convolution_4, squeeze_13, getitem_31, cat, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, getitem_42, convolution_8, squeeze_25, add_46, convolution_9, squeeze_28, add_52, convolution_10, squeeze_31, cat_1, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, getitem_72, convolution_13, squeeze_40, add_74, convolution_14, squeeze_43, add_80, convolution_15, squeeze_46, cat_2, convolution_16, squeeze_49, relu_15, convolution_17, squeeze_52, getitem_102, convolution_18, squeeze_55, getitem_109, convolution_19, squeeze_58, getitem_116, convolution_20, squeeze_61, getitem_123, cat_3, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, getitem_134, convolution_24, squeeze_73, add_133, convolution_25, squeeze_76, add_139, convolution_26, squeeze_79, cat_4, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, getitem_164, convolution_29, squeeze_88, add_161, convolution_30, squeeze_91, add_167, convolution_31, squeeze_94, cat_5, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, getitem_194, convolution_34, squeeze_103, add_189, convolution_35, squeeze_106, add_195, convolution_36, squeeze_109, cat_6, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, getitem_224, convolution_39, squeeze_118, getitem_231, convolution_40, squeeze_121, getitem_238, convolution_41, squeeze_124, getitem_245, cat_7, convolution_42, squeeze_127, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, getitem_256, convolution_45, squeeze_136, add_248, convolution_46, squeeze_139, add_254, convolution_47, squeeze_142, cat_8, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, getitem_286, convolution_50, squeeze_151, add_276, convolution_51, squeeze_154, add_282, convolution_52, squeeze_157, cat_9, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, getitem_316, convolution_55, squeeze_166, add_304, convolution_56, squeeze_169, add_310, convolution_57, squeeze_172, cat_10, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, getitem_346, convolution_60, squeeze_181, add_332, convolution_61, squeeze_184, add_338, convolution_62, squeeze_187, cat_11, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, getitem_376, convolution_65, squeeze_196, add_360, convolution_66, squeeze_199, add_366, convolution_67, squeeze_202, cat_12, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, getitem_406, convolution_70, squeeze_211, getitem_413, convolution_71, squeeze_214, getitem_420, convolution_72, squeeze_217, getitem_427, cat_13, convolution_73, squeeze_220, convolution_74, squeeze_223, relu_70, convolution_75, squeeze_226, getitem_438, convolution_76, squeeze_229, add_419, convolution_77, squeeze_232, add_425, convolution_78, squeeze_235, cat_14, convolution_79, squeeze_238, relu_75, convolution_80, squeeze_241, getitem_468, convolution_81, squeeze_244, add_447, convolution_82, squeeze_247, add_453, convolution_83, squeeze_250, cat_15, convolution_84, squeeze_253, view, permute_1, le, unsqueeze_342, le_1, unsqueeze_354, le_2, unsqueeze_366, le_3, unsqueeze_378, le_4, unsqueeze_390, unsqueeze_402, le_6, unsqueeze_414, le_7, unsqueeze_426, le_8, unsqueeze_438, le_9, unsqueeze_450, unsqueeze_462, unsqueeze_474, le_11, unsqueeze_486, le_12, unsqueeze_498, le_13, unsqueeze_510, le_14, unsqueeze_522, unsqueeze_534, le_16, unsqueeze_546, le_17, unsqueeze_558, le_18, unsqueeze_570, le_19, unsqueeze_582, unsqueeze_594, le_21, unsqueeze_606, le_22, unsqueeze_618, le_23, unsqueeze_630, le_24, unsqueeze_642, unsqueeze_654, le_26, unsqueeze_666, le_27, unsqueeze_678, le_28, unsqueeze_690, le_29, unsqueeze_702, unsqueeze_714, le_31, unsqueeze_726, le_32, unsqueeze_738, le_33, unsqueeze_750, le_34, unsqueeze_762, unsqueeze_774, le_36, unsqueeze_786, le_37, unsqueeze_798, le_38, unsqueeze_810, le_39, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_41, unsqueeze_858, le_42, unsqueeze_870, le_43, unsqueeze_882, le_44, unsqueeze_894, unsqueeze_906, le_46, unsqueeze_918, le_47, unsqueeze_930, le_48, unsqueeze_942, le_49, unsqueeze_954, unsqueeze_966, le_51, unsqueeze_978, le_52, unsqueeze_990, le_53, unsqueeze_1002, le_54, unsqueeze_1014, unsqueeze_1026, le_56, unsqueeze_1038, le_57, unsqueeze_1050, le_58, unsqueeze_1062, le_59, unsqueeze_1074, unsqueeze_1086, unsqueeze_1098, le_61, unsqueeze_1110, le_62, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, le_66, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, le_71, unsqueeze_1230, le_72, unsqueeze_1242, le_73, unsqueeze_1254, le_74, unsqueeze_1266, unsqueeze_1278, unsqueeze_1290, le_76, unsqueeze_1302, le_77, unsqueeze_1314, le_78, unsqueeze_1326, le_79, unsqueeze_1338, unsqueeze_1350, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2next50', benchmark_compiled_module)
