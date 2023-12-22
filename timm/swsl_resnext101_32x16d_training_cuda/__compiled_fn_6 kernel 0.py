
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


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7fusnxlxvprvvstm4oj4ujhdaeinht6vsux72cuiswaoluohf4.py
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (200704*r2)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (200704*r2)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (49*x0) + (200704*r2)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
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


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmubyjcovdxsacf65m6dfhbstzcouo6ak6eb5hybuatzsflde3r.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 4096
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


# kernel path: /tmp/torchinductor_youkaichao/ve/cveont2eqntiwlohwdude2ha4ssrdj77up737jr53ect4nvgkzku.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlyoin5siguhaxgusfj6t3imihi6j2cwpixeerwf5mobkt4mefm.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ui/cuizavtawjqgvm74352uoqjwkyok6t7qbxelbgtj3xev7w7s7hfl.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]

triton_poi_fused_add_div_threshold_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/x4/cx43332npuldyvjtv3pnr22xfwix4qpcmjfbfwc3hkuyns2g2vbi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4y/c4yfhq57xhweihxayiqncdffkfspbovywq6cmiccvwmiljhnyr2j.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/33/c33vb2qlqkd3qzsua66mvygrltyxnwovyfmgsfalxv5adcctmywc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (802816*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (802816*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (802816*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, None)
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lx/clxnjjnkhpwq4eksk2vwbkpqwfbs7xo57nankbew5makpo4dxa77.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 4096
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
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbykt6q4tu766kaof56ipnhliucnm2ktakef6ulwgpbc32hsjgtx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pm/cpmanixza3opdihkxvi6f6eutpequwguibuuyb6egoq7edxcldqh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/75/c75e3dpwanqrywh3k6gj2gmps6r7bjjcg52e54sxqpmmeie7smsn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (401408*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (401408*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (401408*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, None)
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7ppjgq7r57s2voidw2kvfthnl6us3kyrsgvyozmia7euj2p53c.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2048
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
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwa3fn7ivip45bvuxz6k4ejom4prndbo6ocq54sp6t7rvlll3bz.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/e4/ce42t75w76i7fewjwvimhwcqb7uw2qjgcc4exezeei5fbxf5rd56.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_17', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/32/c32sjch3jfwbkdkiqyz75l5ssrbdwbd3kpnlirhcile4md6xdvu7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_18', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4t/c4t3gpgkcafbqr2ghbk4cosuauorms3e47p4nrxihpvdsibpuip4.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/6k/c6kjnkf4xqfkva2t25s4xobbtiszugghxvwgf3jfopt6mdh3ngb4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yw/cywmh3odam6jebqp5cr2t6rh6swtioyqcm3iks5hmobxki37fljw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/52/c522z447rry3rokamdklswn3rc52rvx3mbobzltt4hkkxf5ytaui.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (1605632*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (1605632*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (1605632*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, None)
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/im/cimrqggdt7uh5h362bnd3wrlxrxe3mylvzu4k6aiegbtpklg2e37.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 2048
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


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4x2v75g3aqhneegigm5ire3vsvfpgbnnqjhhazhu3sxi3juurz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hl/chlaqsp5sk3rmwagjkrnhfnrxnalbpymkhnitakbj5qz3ycwh3ka.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4f/c4fkrqyuxnfhhuig2xvrz3hxwztnb45ojiqwx2cvtproakqv5fn6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yw/cywujq35tbkin4qtruzvg7v2gjxbjfyb3r6zrify5qoh37tcc2bn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/34/c344whjr2ukhbkgg646je5qlyvjmbjxxxebbq5zh5qr3zqh2hbms.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_28', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxmg63eucikhyaugriwptwsypn4i2vmtmspfjq6zgqsh7e5ylx2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_29', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxtkt2xy6utc3vrpc6nins34i753aoxzty4mfspbklpyqeksum3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_30', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/to/ctowub3h6epjaq2cyl7zpf5cp774fr2bwvmxml4kckdldggrtupq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_31', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/to/ctoj3zgdu2knrtnjbupofbltxjottoa7hj7uctpqj5w2ypochtc6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_32', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/lk/clklxn5ymxje6xcapybqicm5czsuskla3kybpwot3uejph7wxaso.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (3211264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (3211264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (3211264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kl/ckler2mudau7tfoyv7jiawaujhfcjvr5bde3lu5snwbxtk6qddu6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/3j/c3jx5riptdgxcd5avywmyu2ylx7iowhh3smepfefaebpx2sp2nke.py
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
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5y/c5ya7omzgch7xoxymkgxcknhwu6tr2ua2bmk5bdzcy3kigcam3uk.py
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
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/t2/ct2tbdhfidxhmorwwvwq4yeill3l4cfnz6l6aoiqdctupiq5uxbv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1605632*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (1605632*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (1605632*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzzk65mckhdxz4fbuadqvdyhgkbmge4aznqldxmlowcfcqk4hnd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 512
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


# kernel path: /tmp/torchinductor_youkaichao/tc/ctca3dojoihc657kkal2n2n7iorrxkpvs6gpbw6i4gaywhn3vyke.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6fv6roiqalhxt2id4hasbaqjbcqt6y3pklco3xtpyxxbqh7ync.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_40', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhi5xgy2nemxuiwg3msn44c5rhbfojeb4e4wa7ah7zpqjz6zrcj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_41', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/p6/cp62cj2jrr3rkjrf7qvlq2gnvreqf25n3ogslsx3444l43rt2nhf.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6isuup6buyb353l4hpczclgblthopvusakmouftw5m2m24c7dc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbtp2y4hlybsjlnpf4i6mwboovvsvyj4ywevfjyk53o4u5zpbz2.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_44', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ei/ceihsfvotvid3of2vcxnj4cznmmbriekvgi67gm6cqes2w4yzrol.py
# Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]

triton_poi_fused_add_max_pool2d_with_indices_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_backward_45', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zf/czfgrna7bmpeg2ckiqwc3bn5zmv5glg7ylfw3sqb2mhifkfsglmk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2a/c2ayag4y4advw355gju7fisecyxy53i7p6fe47eb7rcchjtlxiob.py
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
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjnkuclisrlqkpouncaaufpjwhphlbfvqqv46nslenoi5auxn44.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qb/cqb6fte3c5a6ctgdyetaj44ue7iica2xmluz7avyn36glyy72fec.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': ['in_out_ptr0']},
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
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_627, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, relu_42, convolution_46, squeeze_139, relu_43, convolution_47, squeeze_142, relu_44, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, relu_46, convolution_50, squeeze_151, relu_47, convolution_51, squeeze_154, relu_48, convolution_52, squeeze_157, relu_49, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, relu_51, convolution_55, squeeze_166, relu_52, convolution_56, squeeze_169, relu_53, convolution_57, squeeze_172, relu_54, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, relu_56, convolution_60, squeeze_181, relu_57, convolution_61, squeeze_184, relu_58, convolution_62, squeeze_187, relu_59, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, relu_61, convolution_65, squeeze_196, relu_62, convolution_66, squeeze_199, relu_63, convolution_67, squeeze_202, relu_64, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, relu_66, convolution_70, squeeze_211, relu_67, convolution_71, squeeze_214, relu_68, convolution_72, squeeze_217, relu_69, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, relu_71, convolution_75, squeeze_226, relu_72, convolution_76, squeeze_229, relu_73, convolution_77, squeeze_232, relu_74, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, relu_76, convolution_80, squeeze_241, relu_77, convolution_81, squeeze_244, relu_78, convolution_82, squeeze_247, relu_79, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, relu_81, convolution_85, squeeze_256, relu_82, convolution_86, squeeze_259, relu_83, convolution_87, squeeze_262, relu_84, convolution_88, squeeze_265, relu_85, convolution_89, squeeze_268, relu_86, convolution_90, squeeze_271, relu_87, convolution_91, squeeze_274, relu_88, convolution_92, squeeze_277, relu_89, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, relu_91, convolution_95, squeeze_286, relu_92, convolution_96, squeeze_289, convolution_97, squeeze_292, relu_93, convolution_98, squeeze_295, relu_94, convolution_99, squeeze_298, relu_95, convolution_100, squeeze_301, relu_96, convolution_101, squeeze_304, relu_97, convolution_102, squeeze_307, relu_98, convolution_103, squeeze_310, view, permute_1, le, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, unsqueeze_1042, unsqueeze_1054, unsqueeze_1066, unsqueeze_1078, unsqueeze_1090, unsqueeze_1102, unsqueeze_1114, unsqueeze_1126, unsqueeze_1138, unsqueeze_1150, unsqueeze_1162, unsqueeze_1174, unsqueeze_1186, unsqueeze_1198, unsqueeze_1210, unsqueeze_1222, unsqueeze_1234, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, unsqueeze_1282, unsqueeze_1294, unsqueeze_1306, unsqueeze_1318, unsqueeze_1330, unsqueeze_1342, unsqueeze_1354, unsqueeze_1366, unsqueeze_1378, unsqueeze_1390, unsqueeze_1402, unsqueeze_1414, unsqueeze_1426, unsqueeze_1438, unsqueeze_1450, unsqueeze_1462, unsqueeze_1474, unsqueeze_1486, unsqueeze_1498, unsqueeze_1510, unsqueeze_1522, unsqueeze_1534, unsqueeze_1546, unsqueeze_1558, unsqueeze_1570, unsqueeze_1582, unsqueeze_1594, unsqueeze_1606, unsqueeze_1618, unsqueeze_1630, unsqueeze_1642, unsqueeze_1654, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (512, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_7, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_10, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_16, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_19, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_22, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_28, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_31, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_34, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_37, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_38, (1024, ), (1, ))
    assert_size_stride(primals_40, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_46, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_49, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_52, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (1024, ), (1, ))
    assert_size_stride(primals_58, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_61, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_64, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_67, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_70, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_73, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (2048, ), (1, ))
    assert_size_stride(primals_76, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_77, (2048, ), (1, ))
    assert_size_stride(primals_79, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_85, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (2048, ), (1, ))
    assert_size_stride(primals_88, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_89, (2048, ), (1, ))
    assert_size_stride(primals_91, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_94, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (2048, ), (1, ))
    assert_size_stride(primals_97, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_98, (2048, ), (1, ))
    assert_size_stride(primals_100, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_103, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (2048, ), (1, ))
    assert_size_stride(primals_106, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_107, (2048, ), (1, ))
    assert_size_stride(primals_109, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_112, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (2048, ), (1, ))
    assert_size_stride(primals_115, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_116, (2048, ), (1, ))
    assert_size_stride(primals_118, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_121, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (2048, ), (1, ))
    assert_size_stride(primals_124, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_125, (2048, ), (1, ))
    assert_size_stride(primals_127, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_130, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (2048, ), (1, ))
    assert_size_stride(primals_133, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_134, (2048, ), (1, ))
    assert_size_stride(primals_136, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_137, (1024, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_142, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_143, (2048, ), (1, ))
    assert_size_stride(primals_145, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_148, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_151, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_152, (2048, ), (1, ))
    assert_size_stride(primals_154, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_157, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_160, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_161, (2048, ), (1, ))
    assert_size_stride(primals_163, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_166, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_167, (2048, ), (1, ))
    assert_size_stride(primals_169, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_170, (2048, ), (1, ))
    assert_size_stride(primals_172, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_173, (1024, ), (1, ))
    assert_size_stride(primals_175, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_176, (2048, ), (1, ))
    assert_size_stride(primals_178, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_179, (2048, ), (1, ))
    assert_size_stride(primals_181, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_182, (1024, ), (1, ))
    assert_size_stride(primals_184, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_185, (2048, ), (1, ))
    assert_size_stride(primals_187, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_188, (2048, ), (1, ))
    assert_size_stride(primals_190, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_193, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_194, (2048, ), (1, ))
    assert_size_stride(primals_196, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_197, (2048, ), (1, ))
    assert_size_stride(primals_199, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_200, (1024, ), (1, ))
    assert_size_stride(primals_202, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_203, (2048, ), (1, ))
    assert_size_stride(primals_205, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_206, (2048, ), (1, ))
    assert_size_stride(primals_208, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_209, (1024, ), (1, ))
    assert_size_stride(primals_211, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_212, (2048, ), (1, ))
    assert_size_stride(primals_214, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_215, (2048, ), (1, ))
    assert_size_stride(primals_217, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_220, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_221, (2048, ), (1, ))
    assert_size_stride(primals_223, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_226, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_229, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_230, (2048, ), (1, ))
    assert_size_stride(primals_232, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_233, (2048, ), (1, ))
    assert_size_stride(primals_235, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_236, (1024, ), (1, ))
    assert_size_stride(primals_238, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_239, (2048, ), (1, ))
    assert_size_stride(primals_241, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_242, (2048, ), (1, ))
    assert_size_stride(primals_244, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_245, (1024, ), (1, ))
    assert_size_stride(primals_247, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_248, (2048, ), (1, ))
    assert_size_stride(primals_250, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_251, (2048, ), (1, ))
    assert_size_stride(primals_253, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_256, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_257, (2048, ), (1, ))
    assert_size_stride(primals_259, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_260, (2048, ), (1, ))
    assert_size_stride(primals_262, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_263, (1024, ), (1, ))
    assert_size_stride(primals_265, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_266, (2048, ), (1, ))
    assert_size_stride(primals_268, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_269, (2048, ), (1, ))
    assert_size_stride(primals_271, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_272, (1024, ), (1, ))
    assert_size_stride(primals_274, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_275, (2048, ), (1, ))
    assert_size_stride(primals_277, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_278, (2048, ), (1, ))
    assert_size_stride(primals_280, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_283, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_284, (4096, ), (1, ))
    assert_size_stride(primals_286, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_287, (4096, ), (1, ))
    assert_size_stride(primals_289, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_290, (2048, ), (1, ))
    assert_size_stride(primals_292, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_293, (2048, ), (1, ))
    assert_size_stride(primals_295, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_296, (4096, ), (1, ))
    assert_size_stride(primals_298, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_299, (4096, ), (1, ))
    assert_size_stride(primals_301, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_302, (2048, ), (1, ))
    assert_size_stride(primals_304, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_305, (4096, ), (1, ))
    assert_size_stride(primals_307, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_308, (4096, ), (1, ))
    assert_size_stride(primals_310, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_311, (2048, ), (1, ))
    assert_size_stride(primals_627, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(getitem_2, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(getitem_3, (8, 64, 56, 56), (200704, 3136, 56, 1))
    assert_size_stride(convolution_1, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(squeeze_4, (512, ), (1, ))
    assert_size_stride(relu_1, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(convolution_2, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(squeeze_7, (512, ), (1, ))
    assert_size_stride(relu_2, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(convolution_3, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_10, (256, ), (1, ))
    assert_size_stride(convolution_4, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_13, (256, ), (1, ))
    assert_size_stride(relu_3, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_5, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(squeeze_16, (512, ), (1, ))
    assert_size_stride(relu_4, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(convolution_6, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(squeeze_19, (512, ), (1, ))
    assert_size_stride(relu_5, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(convolution_7, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(relu_6, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_8, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(squeeze_25, (512, ), (1, ))
    assert_size_stride(relu_7, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(convolution_9, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(squeeze_28, (512, ), (1, ))
    assert_size_stride(relu_8, (8, 512, 56, 56), (1605632, 3136, 56, 1))
    assert_size_stride(convolution_10, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(relu_9, (8, 256, 56, 56), (802816, 3136, 56, 1))
    assert_size_stride(convolution_11, (8, 1024, 56, 56), (3211264, 3136, 56, 1))
    assert_size_stride(squeeze_34, (1024, ), (1, ))
    assert_size_stride(relu_10, (8, 1024, 56, 56), (3211264, 3136, 56, 1))
    assert_size_stride(convolution_12, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(squeeze_37, (1024, ), (1, ))
    assert_size_stride(relu_11, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(convolution_13, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(convolution_14, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_43, (512, ), (1, ))
    assert_size_stride(relu_12, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_15, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(squeeze_46, (1024, ), (1, ))
    assert_size_stride(relu_13, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(convolution_16, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(squeeze_49, (1024, ), (1, ))
    assert_size_stride(relu_14, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(convolution_17, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_52, (512, ), (1, ))
    assert_size_stride(relu_15, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_18, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(squeeze_55, (1024, ), (1, ))
    assert_size_stride(relu_16, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(convolution_19, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(squeeze_58, (1024, ), (1, ))
    assert_size_stride(relu_17, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(convolution_20, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_61, (512, ), (1, ))
    assert_size_stride(relu_18, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_21, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(squeeze_64, (1024, ), (1, ))
    assert_size_stride(relu_19, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(convolution_22, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(squeeze_67, (1024, ), (1, ))
    assert_size_stride(relu_20, (8, 1024, 28, 28), (802816, 784, 28, 1))
    assert_size_stride(convolution_23, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(squeeze_70, (512, ), (1, ))
    assert_size_stride(relu_21, (8, 512, 28, 28), (401408, 784, 28, 1))
    assert_size_stride(convolution_24, (8, 2048, 28, 28), (1605632, 784, 28, 1))
    assert_size_stride(squeeze_73, (2048, ), (1, ))
    assert_size_stride(relu_22, (8, 2048, 28, 28), (1605632, 784, 28, 1))
    assert_size_stride(convolution_25, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_76, (2048, ), (1, ))
    assert_size_stride(relu_23, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_26, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_79, (1024, ), (1, ))
    assert_size_stride(convolution_27, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_82, (1024, ), (1, ))
    assert_size_stride(relu_24, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_28, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_85, (2048, ), (1, ))
    assert_size_stride(relu_25, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_29, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_88, (2048, ), (1, ))
    assert_size_stride(relu_26, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_30, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_91, (1024, ), (1, ))
    assert_size_stride(relu_27, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_31, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_94, (2048, ), (1, ))
    assert_size_stride(relu_28, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_32, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_97, (2048, ), (1, ))
    assert_size_stride(relu_29, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_33, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_100, (1024, ), (1, ))
    assert_size_stride(relu_30, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_34, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_103, (2048, ), (1, ))
    assert_size_stride(relu_31, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_35, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_106, (2048, ), (1, ))
    assert_size_stride(relu_32, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_36, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_109, (1024, ), (1, ))
    assert_size_stride(relu_33, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_37, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_112, (2048, ), (1, ))
    assert_size_stride(relu_34, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_38, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_115, (2048, ), (1, ))
    assert_size_stride(relu_35, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_39, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_118, (1024, ), (1, ))
    assert_size_stride(relu_36, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_40, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_121, (2048, ), (1, ))
    assert_size_stride(relu_37, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_41, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_124, (2048, ), (1, ))
    assert_size_stride(relu_38, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_42, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_127, (1024, ), (1, ))
    assert_size_stride(relu_39, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_43, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_130, (2048, ), (1, ))
    assert_size_stride(relu_40, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_44, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_133, (2048, ), (1, ))
    assert_size_stride(relu_41, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_45, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_136, (1024, ), (1, ))
    assert_size_stride(relu_42, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_46, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_139, (2048, ), (1, ))
    assert_size_stride(relu_43, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_47, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_142, (2048, ), (1, ))
    assert_size_stride(relu_44, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_48, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_145, (1024, ), (1, ))
    assert_size_stride(relu_45, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_49, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_148, (2048, ), (1, ))
    assert_size_stride(relu_46, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_50, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_151, (2048, ), (1, ))
    assert_size_stride(relu_47, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_51, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_154, (1024, ), (1, ))
    assert_size_stride(relu_48, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_52, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_157, (2048, ), (1, ))
    assert_size_stride(relu_49, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_53, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_160, (2048, ), (1, ))
    assert_size_stride(relu_50, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_54, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_163, (1024, ), (1, ))
    assert_size_stride(relu_51, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_55, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_166, (2048, ), (1, ))
    assert_size_stride(relu_52, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_56, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_169, (2048, ), (1, ))
    assert_size_stride(relu_53, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_57, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_172, (1024, ), (1, ))
    assert_size_stride(relu_54, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_58, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_175, (2048, ), (1, ))
    assert_size_stride(relu_55, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_59, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_178, (2048, ), (1, ))
    assert_size_stride(relu_56, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_60, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_181, (1024, ), (1, ))
    assert_size_stride(relu_57, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_61, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_184, (2048, ), (1, ))
    assert_size_stride(relu_58, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_62, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_187, (2048, ), (1, ))
    assert_size_stride(relu_59, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_63, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_190, (1024, ), (1, ))
    assert_size_stride(relu_60, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_64, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_193, (2048, ), (1, ))
    assert_size_stride(relu_61, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_65, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_196, (2048, ), (1, ))
    assert_size_stride(relu_62, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_66, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_199, (1024, ), (1, ))
    assert_size_stride(relu_63, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_67, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_202, (2048, ), (1, ))
    assert_size_stride(relu_64, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_68, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_205, (2048, ), (1, ))
    assert_size_stride(relu_65, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_69, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_208, (1024, ), (1, ))
    assert_size_stride(relu_66, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_70, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_211, (2048, ), (1, ))
    assert_size_stride(relu_67, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_71, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_214, (2048, ), (1, ))
    assert_size_stride(relu_68, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_72, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_217, (1024, ), (1, ))
    assert_size_stride(relu_69, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_73, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_220, (2048, ), (1, ))
    assert_size_stride(relu_70, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_74, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_223, (2048, ), (1, ))
    assert_size_stride(relu_71, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_75, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_226, (1024, ), (1, ))
    assert_size_stride(relu_72, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_76, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_229, (2048, ), (1, ))
    assert_size_stride(relu_73, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_77, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_232, (2048, ), (1, ))
    assert_size_stride(relu_74, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_78, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_235, (1024, ), (1, ))
    assert_size_stride(relu_75, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_79, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_238, (2048, ), (1, ))
    assert_size_stride(relu_76, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_80, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_241, (2048, ), (1, ))
    assert_size_stride(relu_77, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_81, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_244, (1024, ), (1, ))
    assert_size_stride(relu_78, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_82, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_247, (2048, ), (1, ))
    assert_size_stride(relu_79, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_83, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_250, (2048, ), (1, ))
    assert_size_stride(relu_80, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_84, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_253, (1024, ), (1, ))
    assert_size_stride(relu_81, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_85, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_256, (2048, ), (1, ))
    assert_size_stride(relu_82, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_86, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_259, (2048, ), (1, ))
    assert_size_stride(relu_83, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_87, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_262, (1024, ), (1, ))
    assert_size_stride(relu_84, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_88, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_265, (2048, ), (1, ))
    assert_size_stride(relu_85, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_89, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_268, (2048, ), (1, ))
    assert_size_stride(relu_86, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_90, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_271, (1024, ), (1, ))
    assert_size_stride(relu_87, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_91, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_274, (2048, ), (1, ))
    assert_size_stride(relu_88, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_92, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(squeeze_277, (2048, ), (1, ))
    assert_size_stride(relu_89, (8, 2048, 14, 14), (401408, 196, 14, 1))
    assert_size_stride(convolution_93, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(squeeze_280, (1024, ), (1, ))
    assert_size_stride(relu_90, (8, 1024, 14, 14), (200704, 196, 14, 1))
    assert_size_stride(convolution_94, (8, 4096, 14, 14), (802816, 196, 14, 1))
    assert_size_stride(squeeze_283, (4096, ), (1, ))
    assert_size_stride(relu_91, (8, 4096, 14, 14), (802816, 196, 14, 1))
    assert_size_stride(convolution_95, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(squeeze_286, (4096, ), (1, ))
    assert_size_stride(relu_92, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(convolution_96, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(squeeze_289, (2048, ), (1, ))
    assert_size_stride(convolution_97, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(squeeze_292, (2048, ), (1, ))
    assert_size_stride(relu_93, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(convolution_98, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(squeeze_295, (4096, ), (1, ))
    assert_size_stride(relu_94, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(convolution_99, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(squeeze_298, (4096, ), (1, ))
    assert_size_stride(relu_95, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(convolution_100, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(squeeze_301, (2048, ), (1, ))
    assert_size_stride(relu_96, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(convolution_101, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(squeeze_304, (4096, ), (1, ))
    assert_size_stride(relu_97, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(convolution_102, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(squeeze_307, (4096, ), (1, ))
    assert_size_stride(relu_98, (8, 4096, 7, 7), (200704, 49, 7, 1))
    assert_size_stride(convolution_103, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(squeeze_310, (2048, ), (1, ))
    assert_size_stride(view, (8, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (8, 2048, 7, 7), (100352, 49, 7, 1))
    assert_size_stride(unsqueeze_418, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_586, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_706, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_730, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_754, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_778, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_802, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_826, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_850, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_862, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_874, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_886, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_898, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_910, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_922, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_934, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_946, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_958, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_970, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_982, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_994, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1006, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1018, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1030, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1042, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1054, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1066, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1078, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1090, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1102, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1114, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1126, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1138, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1150, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1162, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1174, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1186, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1198, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1210, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1222, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1234, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1246, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1258, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1270, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1282, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1294, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1306, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1318, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1330, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1342, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1354, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1366, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_1378, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1390, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1402, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1414, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1426, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1438, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1450, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1462, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1474, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1486, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1498, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1510, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1522, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1534, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1546, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1558, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1570, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1582, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1594, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1606, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1618, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1630, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1642, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1654, (1, 64, 1, 1), (64, 1, 1, 1))
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
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_103, unsqueeze_418, squeeze_310, buf3, buf4, buf5, 2048, 392, grid=grid(2048), stream=stream0)
        buf6 = empty((8, 2048, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_2.run(le, buf0, convolution_103, unsqueeze_418, buf4, squeeze_310, buf3, primals_311, buf6, 802816, grid=grid(802816), stream=stream0)
        del convolution_103
        del primals_311
        del squeeze_310
        del unsqueeze_418
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf7 = aten.convolution_backward(buf6, relu_98, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_310
        buf8 = buf7[0]
        buf9 = buf7[1]
        del buf7
        buf10 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf11 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf12 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(relu_98, buf8, convolution_102, unsqueeze_430, squeeze_307, buf10, buf11, buf12, 4096, 392, grid=grid(4096), stream=stream0)
        buf13 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(buf13, relu_98, convolution_102, unsqueeze_430, buf11, squeeze_307, buf10, primals_308, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_102
        del primals_308
        del relu_98
        del squeeze_307
        del unsqueeze_430
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf14 = aten.convolution_backward(buf13, relu_97, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf13
        del primals_307
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = buf11; del buf11  # reuse
        buf18 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf19 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(relu_97, buf15, convolution_101, unsqueeze_442, squeeze_304, buf17, buf18, buf19, 4096, 392, grid=grid(4096), stream=stream0)
        buf20 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(buf20, relu_97, convolution_101, unsqueeze_442, buf18, squeeze_304, buf17, primals_305, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_101
        del primals_305
        del relu_97
        del squeeze_304
        del unsqueeze_442
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf21 = aten.convolution_backward(buf20, relu_96, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf20
        del primals_304
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = buf4; del buf4  # reuse
        buf25 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf27 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_native_batch_norm_backward_threshold_backward_5.run(relu_96, le, buf0, buf22, convolution_100, unsqueeze_454, squeeze_301, buf24, buf25, buf27, 2048, 392, grid=grid(2048), stream=stream0)
        buf26 = buf6; del buf6  # reuse
        buf28 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6.run(buf28, relu_96, le, buf0, buf22, convolution_100, unsqueeze_454, buf25, squeeze_301, buf24, primals_302, 802816, grid=grid(802816), stream=stream0)
        del convolution_100
        del primals_302
        del squeeze_301
        del unsqueeze_454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf29 = aten.convolution_backward(buf28, relu_95, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_301
        buf30 = buf29[0]
        buf31 = buf29[1]
        del buf29
        buf32 = buf18; del buf18  # reuse
        buf33 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf34 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(relu_95, buf30, convolution_99, unsqueeze_466, squeeze_298, buf32, buf33, buf34, 4096, 392, grid=grid(4096), stream=stream0)
        buf35 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(buf35, relu_95, convolution_99, unsqueeze_466, buf33, squeeze_298, buf32, primals_299, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_99
        del primals_299
        del relu_95
        del squeeze_298
        del unsqueeze_466
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf36 = aten.convolution_backward(buf35, relu_94, primals_298, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf35
        del primals_298
        buf37 = buf36[0]
        buf38 = buf36[1]
        del buf36
        buf39 = buf33; del buf33  # reuse
        buf40 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf41 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(relu_94, buf37, convolution_98, unsqueeze_478, squeeze_295, buf39, buf40, buf41, 4096, 392, grid=grid(4096), stream=stream0)
        buf42 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(buf42, relu_94, convolution_98, unsqueeze_478, buf40, squeeze_295, buf39, primals_296, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_98
        del primals_296
        del relu_94
        del squeeze_295
        del unsqueeze_478
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf43 = aten.convolution_backward(buf42, relu_93, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf42
        del primals_295
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        buf46 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
        triton_poi_fused_add_div_threshold_backward_7.run(buf46, relu_93, relu_96, le, buf0, buf44, 802816, grid=grid(802816), stream=stream0)
        del buf0
        del le
        del relu_93
        del relu_96
        buf47 = buf25; del buf25  # reuse
        buf48 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf54 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf49 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf55 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_8.run(buf46, convolution_97, unsqueeze_490, convolution_96, unsqueeze_502, squeeze_292, squeeze_289, buf47, buf48, buf54, buf49, buf55, 2048, 392, grid=grid(2048), stream=stream0)
        buf50 = buf44; del buf44  # reuse
        buf56 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_9.run(buf46, convolution_97, unsqueeze_490, buf48, squeeze_292, buf47, primals_293, convolution_96, unsqueeze_502, buf54, squeeze_289, primals_290, buf50, buf56, 802816, grid=grid(802816), stream=stream0)
        del buf46
        del convolution_96
        del convolution_97
        del primals_290
        del primals_293
        del squeeze_289
        del squeeze_292
        del unsqueeze_490
        del unsqueeze_502
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf51 = aten.convolution_backward(buf50, relu_90, primals_292, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf50
        del primals_292
        buf52 = buf51[0]
        buf53 = buf51[1]
        del buf51
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf57 = aten.convolution_backward(buf56, relu_92, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf56
        del primals_289
        buf58 = buf57[0]
        buf59 = buf57[1]
        del buf57
        buf60 = buf40; del buf40  # reuse
        buf61 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf62 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(relu_92, buf58, convolution_95, unsqueeze_514, squeeze_286, buf60, buf61, buf62, 4096, 392, grid=grid(4096), stream=stream0)
        buf63 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(buf63, relu_92, convolution_95, unsqueeze_514, buf61, squeeze_286, buf60, primals_287, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_95
        del primals_287
        del relu_92
        del squeeze_286
        del unsqueeze_514
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf64 = aten.convolution_backward(buf63, relu_91, primals_286, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_286
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf61; del buf61  # reuse
        buf68 = empty((4096, ), device='cuda', dtype=torch.float32)
        buf69 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_10.run(relu_91, buf65, convolution_94, unsqueeze_526, squeeze_283, buf67, buf68, buf69, 4096, 1568, grid=grid(4096), stream=stream0)
        buf70 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11.run(buf70, relu_91, convolution_94, unsqueeze_526, buf68, squeeze_283, buf67, primals_284, 6422528, grid=grid(6422528), stream=stream0)
        del buf68
        del convolution_94
        del primals_284
        del relu_91
        del squeeze_283
        del unsqueeze_526
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf71 = aten.convolution_backward(buf70, relu_90, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf70
        del primals_283
        buf72 = buf71[0]
        buf73 = buf71[1]
        del buf71
        buf74 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf75 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf77 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_90, buf52, buf72, convolution_93, unsqueeze_538, squeeze_280, buf74, buf75, buf77, 1024, 1568, grid=grid(1024), stream=stream0)
        buf76 = reinterpret_tensor(buf63, (8, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_90, buf52, buf72, convolution_93, unsqueeze_538, buf75, squeeze_280, buf74, primals_281, buf76, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_93
        del primals_281
        del squeeze_280
        del unsqueeze_538
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf78 = aten.convolution_backward(buf76, relu_89, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf76
        del primals_280
        buf79 = buf78[0]
        buf80 = buf78[1]
        del buf78
        buf81 = buf54; del buf54  # reuse
        buf82 = buf48; del buf48  # reuse
        buf83 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_89, buf79, convolution_92, unsqueeze_550, squeeze_277, buf81, buf82, buf83, 2048, 1568, grid=grid(2048), stream=stream0)
        buf84 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf84, relu_89, convolution_92, unsqueeze_550, buf82, squeeze_277, buf81, primals_278, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_92
        del primals_278
        del relu_89
        del squeeze_277
        del unsqueeze_550
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf85 = aten.convolution_backward(buf84, relu_88, primals_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf84
        del primals_277
        buf86 = buf85[0]
        buf87 = buf85[1]
        del buf85
        buf88 = buf82; del buf82  # reuse
        buf89 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf90 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_88, buf86, convolution_91, unsqueeze_562, squeeze_274, buf88, buf89, buf90, 2048, 1568, grid=grid(2048), stream=stream0)
        buf91 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf91, relu_88, convolution_91, unsqueeze_562, buf89, squeeze_274, buf88, primals_275, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_91
        del primals_275
        del relu_88
        del squeeze_274
        del unsqueeze_562
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf92 = aten.convolution_backward(buf91, relu_87, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf91
        del primals_274
        buf93 = buf92[0]
        buf94 = buf92[1]
        del buf92
        buf95 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf95, relu_87, relu_90, buf72, buf93, 1605632, grid=grid(1605632), stream=stream0)
        del buf72
        del relu_87
        del relu_90
        buf96 = buf75; del buf75  # reuse
        buf97 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf98 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf95, convolution_90, unsqueeze_574, squeeze_271, buf96, buf97, buf98, 1024, 1568, grid=grid(1024), stream=stream0)
        buf99 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf95, convolution_90, unsqueeze_574, buf97, squeeze_271, buf96, primals_272, buf99, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_90
        del primals_272
        del squeeze_271
        del unsqueeze_574
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf100 = aten.convolution_backward(buf99, relu_86, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_271
        buf101 = buf100[0]
        buf102 = buf100[1]
        del buf100
        buf103 = buf89; del buf89  # reuse
        buf104 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf105 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_86, buf101, convolution_89, unsqueeze_586, squeeze_268, buf103, buf104, buf105, 2048, 1568, grid=grid(2048), stream=stream0)
        buf106 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf106, relu_86, convolution_89, unsqueeze_586, buf104, squeeze_268, buf103, primals_269, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_89
        del primals_269
        del relu_86
        del squeeze_268
        del unsqueeze_586
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf107 = aten.convolution_backward(buf106, relu_85, primals_268, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf106
        del primals_268
        buf108 = buf107[0]
        buf109 = buf107[1]
        del buf107
        buf110 = buf104; del buf104  # reuse
        buf111 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf112 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_85, buf108, convolution_88, unsqueeze_598, squeeze_265, buf110, buf111, buf112, 2048, 1568, grid=grid(2048), stream=stream0)
        buf113 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf113, relu_85, convolution_88, unsqueeze_598, buf111, squeeze_265, buf110, primals_266, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_88
        del primals_266
        del relu_85
        del squeeze_265
        del unsqueeze_598
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf114 = aten.convolution_backward(buf113, relu_84, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf113
        del primals_265
        buf115 = buf114[0]
        buf116 = buf114[1]
        del buf114
        buf117 = buf97; del buf97  # reuse
        buf118 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf120 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_84, buf95, buf115, convolution_87, unsqueeze_610, squeeze_262, buf117, buf118, buf120, 1024, 1568, grid=grid(1024), stream=stream0)
        buf119 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_84, buf95, buf115, convolution_87, unsqueeze_610, buf118, squeeze_262, buf117, primals_263, buf119, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_87
        del primals_263
        del squeeze_262
        del unsqueeze_610
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf121 = aten.convolution_backward(buf119, relu_83, primals_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf119
        del primals_262
        buf122 = buf121[0]
        buf123 = buf121[1]
        del buf121
        buf124 = buf111; del buf111  # reuse
        buf125 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf126 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_83, buf122, convolution_86, unsqueeze_622, squeeze_259, buf124, buf125, buf126, 2048, 1568, grid=grid(2048), stream=stream0)
        buf127 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf127, relu_83, convolution_86, unsqueeze_622, buf125, squeeze_259, buf124, primals_260, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_86
        del primals_260
        del relu_83
        del squeeze_259
        del unsqueeze_622
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf128 = aten.convolution_backward(buf127, relu_82, primals_259, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf127
        del primals_259
        buf129 = buf128[0]
        buf130 = buf128[1]
        del buf128
        buf131 = buf125; del buf125  # reuse
        buf132 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf133 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_82, buf129, convolution_85, unsqueeze_634, squeeze_256, buf131, buf132, buf133, 2048, 1568, grid=grid(2048), stream=stream0)
        buf134 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf134, relu_82, convolution_85, unsqueeze_634, buf132, squeeze_256, buf131, primals_257, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_85
        del primals_257
        del relu_82
        del squeeze_256
        del unsqueeze_634
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf135 = aten.convolution_backward(buf134, relu_81, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf134
        del primals_256
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        buf138 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_19.run(buf138, relu_81, relu_84, buf95, buf136, 1605632, grid=grid(1605632), stream=stream0)
        del buf136
        del relu_81
        del relu_84
        buf139 = buf118; del buf118  # reuse
        buf140 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf141 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf138, convolution_84, unsqueeze_646, squeeze_253, buf139, buf140, buf141, 1024, 1568, grid=grid(1024), stream=stream0)
        buf142 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf138, convolution_84, unsqueeze_646, buf140, squeeze_253, buf139, primals_254, buf142, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_84
        del primals_254
        del squeeze_253
        del unsqueeze_646
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf143 = aten.convolution_backward(buf142, relu_80, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_253
        buf144 = buf143[0]
        buf145 = buf143[1]
        del buf143
        buf146 = buf132; del buf132  # reuse
        buf147 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf148 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_80, buf144, convolution_83, unsqueeze_658, squeeze_250, buf146, buf147, buf148, 2048, 1568, grid=grid(2048), stream=stream0)
        buf149 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf149, relu_80, convolution_83, unsqueeze_658, buf147, squeeze_250, buf146, primals_251, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_83
        del primals_251
        del relu_80
        del squeeze_250
        del unsqueeze_658
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf150 = aten.convolution_backward(buf149, relu_79, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf149
        del primals_250
        buf151 = buf150[0]
        buf152 = buf150[1]
        del buf150
        buf153 = buf147; del buf147  # reuse
        buf154 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf155 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_79, buf151, convolution_82, unsqueeze_670, squeeze_247, buf153, buf154, buf155, 2048, 1568, grid=grid(2048), stream=stream0)
        buf156 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf156, relu_79, convolution_82, unsqueeze_670, buf154, squeeze_247, buf153, primals_248, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_82
        del primals_248
        del relu_79
        del squeeze_247
        del unsqueeze_670
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf157 = aten.convolution_backward(buf156, relu_78, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf156
        del primals_247
        buf158 = buf157[0]
        buf159 = buf157[1]
        del buf157
        buf160 = buf140; del buf140  # reuse
        buf161 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf163 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_78, buf138, buf158, convolution_81, unsqueeze_682, squeeze_244, buf160, buf161, buf163, 1024, 1568, grid=grid(1024), stream=stream0)
        buf162 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_78, buf138, buf158, convolution_81, unsqueeze_682, buf161, squeeze_244, buf160, primals_245, buf162, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_81
        del primals_245
        del squeeze_244
        del unsqueeze_682
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf164 = aten.convolution_backward(buf162, relu_77, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf162
        del primals_244
        buf165 = buf164[0]
        buf166 = buf164[1]
        del buf164
        buf167 = buf154; del buf154  # reuse
        buf168 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf169 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_77, buf165, convolution_80, unsqueeze_694, squeeze_241, buf167, buf168, buf169, 2048, 1568, grid=grid(2048), stream=stream0)
        buf170 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf170, relu_77, convolution_80, unsqueeze_694, buf168, squeeze_241, buf167, primals_242, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_80
        del primals_242
        del relu_77
        del squeeze_241
        del unsqueeze_694
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf171 = aten.convolution_backward(buf170, relu_76, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf170
        del primals_241
        buf172 = buf171[0]
        buf173 = buf171[1]
        del buf171
        buf174 = buf168; del buf168  # reuse
        buf175 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf176 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_76, buf172, convolution_79, unsqueeze_706, squeeze_238, buf174, buf175, buf176, 2048, 1568, grid=grid(2048), stream=stream0)
        buf177 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf177, relu_76, convolution_79, unsqueeze_706, buf175, squeeze_238, buf174, primals_239, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_79
        del primals_239
        del relu_76
        del squeeze_238
        del unsqueeze_706
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf178 = aten.convolution_backward(buf177, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf177
        del primals_238
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf181 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf181, relu_75, relu_78, buf158, buf179, 1605632, grid=grid(1605632), stream=stream0)
        del buf158
        del relu_75
        del relu_78
        buf182 = buf161; del buf161  # reuse
        buf183 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf184 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf181, convolution_78, unsqueeze_718, squeeze_235, buf182, buf183, buf184, 1024, 1568, grid=grid(1024), stream=stream0)
        buf185 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf181, convolution_78, unsqueeze_718, buf183, squeeze_235, buf182, primals_236, buf185, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_78
        del primals_236
        del squeeze_235
        del unsqueeze_718
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf186 = aten.convolution_backward(buf185, relu_74, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_235
        buf187 = buf186[0]
        buf188 = buf186[1]
        del buf186
        buf189 = buf175; del buf175  # reuse
        buf190 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf191 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_74, buf187, convolution_77, unsqueeze_730, squeeze_232, buf189, buf190, buf191, 2048, 1568, grid=grid(2048), stream=stream0)
        buf192 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf192, relu_74, convolution_77, unsqueeze_730, buf190, squeeze_232, buf189, primals_233, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_77
        del primals_233
        del relu_74
        del squeeze_232
        del unsqueeze_730
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf193 = aten.convolution_backward(buf192, relu_73, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf192
        del primals_232
        buf194 = buf193[0]
        buf195 = buf193[1]
        del buf193
        buf196 = buf190; del buf190  # reuse
        buf197 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf198 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_73, buf194, convolution_76, unsqueeze_742, squeeze_229, buf196, buf197, buf198, 2048, 1568, grid=grid(2048), stream=stream0)
        buf199 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf199, relu_73, convolution_76, unsqueeze_742, buf197, squeeze_229, buf196, primals_230, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_76
        del primals_230
        del relu_73
        del squeeze_229
        del unsqueeze_742
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf200 = aten.convolution_backward(buf199, relu_72, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf199
        del primals_229
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = buf183; del buf183  # reuse
        buf204 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf206 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_72, buf181, buf201, convolution_75, unsqueeze_754, squeeze_226, buf203, buf204, buf206, 1024, 1568, grid=grid(1024), stream=stream0)
        buf205 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_72, buf181, buf201, convolution_75, unsqueeze_754, buf204, squeeze_226, buf203, primals_227, buf205, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_75
        del primals_227
        del squeeze_226
        del unsqueeze_754
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf207 = aten.convolution_backward(buf205, relu_71, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf205
        del primals_226
        buf208 = buf207[0]
        buf209 = buf207[1]
        del buf207
        buf210 = buf197; del buf197  # reuse
        buf211 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf212 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_71, buf208, convolution_74, unsqueeze_766, squeeze_223, buf210, buf211, buf212, 2048, 1568, grid=grid(2048), stream=stream0)
        buf213 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf213, relu_71, convolution_74, unsqueeze_766, buf211, squeeze_223, buf210, primals_224, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_74
        del primals_224
        del relu_71
        del squeeze_223
        del unsqueeze_766
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf214 = aten.convolution_backward(buf213, relu_70, primals_223, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf213
        del primals_223
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        buf217 = buf211; del buf211  # reuse
        buf218 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf219 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_70, buf215, convolution_73, unsqueeze_778, squeeze_220, buf217, buf218, buf219, 2048, 1568, grid=grid(2048), stream=stream0)
        buf220 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf220, relu_70, convolution_73, unsqueeze_778, buf218, squeeze_220, buf217, primals_221, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_73
        del primals_221
        del relu_70
        del squeeze_220
        del unsqueeze_778
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf221 = aten.convolution_backward(buf220, relu_69, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf220
        del primals_220
        buf222 = buf221[0]
        buf223 = buf221[1]
        del buf221
        buf224 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf224, relu_69, relu_72, buf201, buf222, 1605632, grid=grid(1605632), stream=stream0)
        del buf201
        del relu_69
        del relu_72
        buf225 = buf204; del buf204  # reuse
        buf226 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf227 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf224, convolution_72, unsqueeze_790, squeeze_217, buf225, buf226, buf227, 1024, 1568, grid=grid(1024), stream=stream0)
        buf228 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf224, convolution_72, unsqueeze_790, buf226, squeeze_217, buf225, primals_218, buf228, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_72
        del primals_218
        del squeeze_217
        del unsqueeze_790
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf229 = aten.convolution_backward(buf228, relu_68, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_217
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf232 = buf218; del buf218  # reuse
        buf233 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf234 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_68, buf230, convolution_71, unsqueeze_802, squeeze_214, buf232, buf233, buf234, 2048, 1568, grid=grid(2048), stream=stream0)
        buf235 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf235, relu_68, convolution_71, unsqueeze_802, buf233, squeeze_214, buf232, primals_215, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_71
        del primals_215
        del relu_68
        del squeeze_214
        del unsqueeze_802
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf236 = aten.convolution_backward(buf235, relu_67, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf235
        del primals_214
        buf237 = buf236[0]
        buf238 = buf236[1]
        del buf236
        buf239 = buf233; del buf233  # reuse
        buf240 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf241 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_67, buf237, convolution_70, unsqueeze_814, squeeze_211, buf239, buf240, buf241, 2048, 1568, grid=grid(2048), stream=stream0)
        buf242 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf242, relu_67, convolution_70, unsqueeze_814, buf240, squeeze_211, buf239, primals_212, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_70
        del primals_212
        del relu_67
        del squeeze_211
        del unsqueeze_814
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf243 = aten.convolution_backward(buf242, relu_66, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf242
        del primals_211
        buf244 = buf243[0]
        buf245 = buf243[1]
        del buf243
        buf246 = buf226; del buf226  # reuse
        buf247 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf249 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_66, buf224, buf244, convolution_69, unsqueeze_826, squeeze_208, buf246, buf247, buf249, 1024, 1568, grid=grid(1024), stream=stream0)
        buf248 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_66, buf224, buf244, convolution_69, unsqueeze_826, buf247, squeeze_208, buf246, primals_209, buf248, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_69
        del primals_209
        del squeeze_208
        del unsqueeze_826
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf250 = aten.convolution_backward(buf248, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf248
        del primals_208
        buf251 = buf250[0]
        buf252 = buf250[1]
        del buf250
        buf253 = buf240; del buf240  # reuse
        buf254 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf255 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_65, buf251, convolution_68, unsqueeze_838, squeeze_205, buf253, buf254, buf255, 2048, 1568, grid=grid(2048), stream=stream0)
        buf256 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf256, relu_65, convolution_68, unsqueeze_838, buf254, squeeze_205, buf253, primals_206, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_68
        del primals_206
        del relu_65
        del squeeze_205
        del unsqueeze_838
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf257 = aten.convolution_backward(buf256, relu_64, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf256
        del primals_205
        buf258 = buf257[0]
        buf259 = buf257[1]
        del buf257
        buf260 = buf254; del buf254  # reuse
        buf261 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf262 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_64, buf258, convolution_67, unsqueeze_850, squeeze_202, buf260, buf261, buf262, 2048, 1568, grid=grid(2048), stream=stream0)
        buf263 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf263, relu_64, convolution_67, unsqueeze_850, buf261, squeeze_202, buf260, primals_203, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_67
        del primals_203
        del relu_64
        del squeeze_202
        del unsqueeze_850
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf264 = aten.convolution_backward(buf263, relu_63, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf263
        del primals_202
        buf265 = buf264[0]
        buf266 = buf264[1]
        del buf264
        buf267 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf267, relu_63, relu_66, buf244, buf265, 1605632, grid=grid(1605632), stream=stream0)
        del buf244
        del relu_63
        del relu_66
        buf268 = buf247; del buf247  # reuse
        buf269 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf270 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf267, convolution_66, unsqueeze_862, squeeze_199, buf268, buf269, buf270, 1024, 1568, grid=grid(1024), stream=stream0)
        buf271 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf267, convolution_66, unsqueeze_862, buf269, squeeze_199, buf268, primals_200, buf271, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_66
        del primals_200
        del squeeze_199
        del unsqueeze_862
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf272 = aten.convolution_backward(buf271, relu_62, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_199
        buf273 = buf272[0]
        buf274 = buf272[1]
        del buf272
        buf275 = buf261; del buf261  # reuse
        buf276 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf277 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_62, buf273, convolution_65, unsqueeze_874, squeeze_196, buf275, buf276, buf277, 2048, 1568, grid=grid(2048), stream=stream0)
        buf278 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf278, relu_62, convolution_65, unsqueeze_874, buf276, squeeze_196, buf275, primals_197, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_65
        del primals_197
        del relu_62
        del squeeze_196
        del unsqueeze_874
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf279 = aten.convolution_backward(buf278, relu_61, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf278
        del primals_196
        buf280 = buf279[0]
        buf281 = buf279[1]
        del buf279
        buf282 = buf276; del buf276  # reuse
        buf283 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf284 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_61, buf280, convolution_64, unsqueeze_886, squeeze_193, buf282, buf283, buf284, 2048, 1568, grid=grid(2048), stream=stream0)
        buf285 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf285, relu_61, convolution_64, unsqueeze_886, buf283, squeeze_193, buf282, primals_194, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_64
        del primals_194
        del relu_61
        del squeeze_193
        del unsqueeze_886
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf286 = aten.convolution_backward(buf285, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf285
        del primals_193
        buf287 = buf286[0]
        buf288 = buf286[1]
        del buf286
        buf289 = buf269; del buf269  # reuse
        buf290 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf292 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_60, buf267, buf287, convolution_63, unsqueeze_898, squeeze_190, buf289, buf290, buf292, 1024, 1568, grid=grid(1024), stream=stream0)
        buf291 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_60, buf267, buf287, convolution_63, unsqueeze_898, buf290, squeeze_190, buf289, primals_191, buf291, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_63
        del primals_191
        del squeeze_190
        del unsqueeze_898
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf293 = aten.convolution_backward(buf291, relu_59, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf291
        del primals_190
        buf294 = buf293[0]
        buf295 = buf293[1]
        del buf293
        buf296 = buf283; del buf283  # reuse
        buf297 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf298 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_59, buf294, convolution_62, unsqueeze_910, squeeze_187, buf296, buf297, buf298, 2048, 1568, grid=grid(2048), stream=stream0)
        buf299 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf299, relu_59, convolution_62, unsqueeze_910, buf297, squeeze_187, buf296, primals_188, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_62
        del primals_188
        del relu_59
        del squeeze_187
        del unsqueeze_910
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf300 = aten.convolution_backward(buf299, relu_58, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf299
        del primals_187
        buf301 = buf300[0]
        buf302 = buf300[1]
        del buf300
        buf303 = buf297; del buf297  # reuse
        buf304 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf305 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_58, buf301, convolution_61, unsqueeze_922, squeeze_184, buf303, buf304, buf305, 2048, 1568, grid=grid(2048), stream=stream0)
        buf306 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf306, relu_58, convolution_61, unsqueeze_922, buf304, squeeze_184, buf303, primals_185, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_61
        del primals_185
        del relu_58
        del squeeze_184
        del unsqueeze_922
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf307 = aten.convolution_backward(buf306, relu_57, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf306
        del primals_184
        buf308 = buf307[0]
        buf309 = buf307[1]
        del buf307
        buf310 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf310, relu_57, relu_60, buf287, buf308, 1605632, grid=grid(1605632), stream=stream0)
        del buf287
        del relu_57
        del relu_60
        buf311 = buf290; del buf290  # reuse
        buf312 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf313 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf310, convolution_60, unsqueeze_934, squeeze_181, buf311, buf312, buf313, 1024, 1568, grid=grid(1024), stream=stream0)
        buf314 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf310, convolution_60, unsqueeze_934, buf312, squeeze_181, buf311, primals_182, buf314, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_60
        del primals_182
        del squeeze_181
        del unsqueeze_934
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf315 = aten.convolution_backward(buf314, relu_56, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_181
        buf316 = buf315[0]
        buf317 = buf315[1]
        del buf315
        buf318 = buf304; del buf304  # reuse
        buf319 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf320 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_56, buf316, convolution_59, unsqueeze_946, squeeze_178, buf318, buf319, buf320, 2048, 1568, grid=grid(2048), stream=stream0)
        buf321 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf321, relu_56, convolution_59, unsqueeze_946, buf319, squeeze_178, buf318, primals_179, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_59
        del primals_179
        del relu_56
        del squeeze_178
        del unsqueeze_946
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf322 = aten.convolution_backward(buf321, relu_55, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf321
        del primals_178
        buf323 = buf322[0]
        buf324 = buf322[1]
        del buf322
        buf325 = buf319; del buf319  # reuse
        buf326 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf327 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_55, buf323, convolution_58, unsqueeze_958, squeeze_175, buf325, buf326, buf327, 2048, 1568, grid=grid(2048), stream=stream0)
        buf328 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf328, relu_55, convolution_58, unsqueeze_958, buf326, squeeze_175, buf325, primals_176, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_58
        del primals_176
        del relu_55
        del squeeze_175
        del unsqueeze_958
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf329 = aten.convolution_backward(buf328, relu_54, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf328
        del primals_175
        buf330 = buf329[0]
        buf331 = buf329[1]
        del buf329
        buf332 = buf312; del buf312  # reuse
        buf333 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf335 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_54, buf310, buf330, convolution_57, unsqueeze_970, squeeze_172, buf332, buf333, buf335, 1024, 1568, grid=grid(1024), stream=stream0)
        buf334 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_54, buf310, buf330, convolution_57, unsqueeze_970, buf333, squeeze_172, buf332, primals_173, buf334, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_57
        del primals_173
        del squeeze_172
        del unsqueeze_970
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf336 = aten.convolution_backward(buf334, relu_53, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf334
        del primals_172
        buf337 = buf336[0]
        buf338 = buf336[1]
        del buf336
        buf339 = buf326; del buf326  # reuse
        buf340 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf341 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_53, buf337, convolution_56, unsqueeze_982, squeeze_169, buf339, buf340, buf341, 2048, 1568, grid=grid(2048), stream=stream0)
        buf342 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf342, relu_53, convolution_56, unsqueeze_982, buf340, squeeze_169, buf339, primals_170, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_56
        del primals_170
        del relu_53
        del squeeze_169
        del unsqueeze_982
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf343 = aten.convolution_backward(buf342, relu_52, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf342
        del primals_169
        buf344 = buf343[0]
        buf345 = buf343[1]
        del buf343
        buf346 = buf340; del buf340  # reuse
        buf347 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf348 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_52, buf344, convolution_55, unsqueeze_994, squeeze_166, buf346, buf347, buf348, 2048, 1568, grid=grid(2048), stream=stream0)
        buf349 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf349, relu_52, convolution_55, unsqueeze_994, buf347, squeeze_166, buf346, primals_167, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_55
        del primals_167
        del relu_52
        del squeeze_166
        del unsqueeze_994
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf350 = aten.convolution_backward(buf349, relu_51, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf349
        del primals_166
        buf351 = buf350[0]
        buf352 = buf350[1]
        del buf350
        buf353 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf353, relu_51, relu_54, buf330, buf351, 1605632, grid=grid(1605632), stream=stream0)
        del buf330
        del relu_51
        del relu_54
        buf354 = buf333; del buf333  # reuse
        buf355 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf356 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf353, convolution_54, unsqueeze_1006, squeeze_163, buf354, buf355, buf356, 1024, 1568, grid=grid(1024), stream=stream0)
        buf357 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf353, convolution_54, unsqueeze_1006, buf355, squeeze_163, buf354, primals_164, buf357, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_54
        del primals_164
        del squeeze_163
        del unsqueeze_1006
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf358 = aten.convolution_backward(buf357, relu_50, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_163
        buf359 = buf358[0]
        buf360 = buf358[1]
        del buf358
        buf361 = buf347; del buf347  # reuse
        buf362 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf363 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_50, buf359, convolution_53, unsqueeze_1018, squeeze_160, buf361, buf362, buf363, 2048, 1568, grid=grid(2048), stream=stream0)
        buf364 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf364, relu_50, convolution_53, unsqueeze_1018, buf362, squeeze_160, buf361, primals_161, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_53
        del primals_161
        del relu_50
        del squeeze_160
        del unsqueeze_1018
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf365 = aten.convolution_backward(buf364, relu_49, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf364
        del primals_160
        buf366 = buf365[0]
        buf367 = buf365[1]
        del buf365
        buf368 = buf362; del buf362  # reuse
        buf369 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf370 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_49, buf366, convolution_52, unsqueeze_1030, squeeze_157, buf368, buf369, buf370, 2048, 1568, grid=grid(2048), stream=stream0)
        buf371 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf371, relu_49, convolution_52, unsqueeze_1030, buf369, squeeze_157, buf368, primals_158, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_52
        del primals_158
        del relu_49
        del squeeze_157
        del unsqueeze_1030
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf372 = aten.convolution_backward(buf371, relu_48, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf371
        del primals_157
        buf373 = buf372[0]
        buf374 = buf372[1]
        del buf372
        buf375 = buf355; del buf355  # reuse
        buf376 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf378 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_48, buf353, buf373, convolution_51, unsqueeze_1042, squeeze_154, buf375, buf376, buf378, 1024, 1568, grid=grid(1024), stream=stream0)
        buf377 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_48, buf353, buf373, convolution_51, unsqueeze_1042, buf376, squeeze_154, buf375, primals_155, buf377, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_51
        del primals_155
        del squeeze_154
        del unsqueeze_1042
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf379 = aten.convolution_backward(buf377, relu_47, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf377
        del primals_154
        buf380 = buf379[0]
        buf381 = buf379[1]
        del buf379
        buf382 = buf369; del buf369  # reuse
        buf383 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf384 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_47, buf380, convolution_50, unsqueeze_1054, squeeze_151, buf382, buf383, buf384, 2048, 1568, grid=grid(2048), stream=stream0)
        buf385 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf385, relu_47, convolution_50, unsqueeze_1054, buf383, squeeze_151, buf382, primals_152, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_50
        del primals_152
        del relu_47
        del squeeze_151
        del unsqueeze_1054
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf386 = aten.convolution_backward(buf385, relu_46, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf385
        del primals_151
        buf387 = buf386[0]
        buf388 = buf386[1]
        del buf386
        buf389 = buf383; del buf383  # reuse
        buf390 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf391 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_46, buf387, convolution_49, unsqueeze_1066, squeeze_148, buf389, buf390, buf391, 2048, 1568, grid=grid(2048), stream=stream0)
        buf392 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf392, relu_46, convolution_49, unsqueeze_1066, buf390, squeeze_148, buf389, primals_149, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_49
        del primals_149
        del relu_46
        del squeeze_148
        del unsqueeze_1066
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf393 = aten.convolution_backward(buf392, relu_45, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf392
        del primals_148
        buf394 = buf393[0]
        buf395 = buf393[1]
        del buf393
        buf396 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf396, relu_45, relu_48, buf373, buf394, 1605632, grid=grid(1605632), stream=stream0)
        del buf373
        del relu_45
        del relu_48
        buf397 = buf376; del buf376  # reuse
        buf398 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf399 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf396, convolution_48, unsqueeze_1078, squeeze_145, buf397, buf398, buf399, 1024, 1568, grid=grid(1024), stream=stream0)
        buf400 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf396, convolution_48, unsqueeze_1078, buf398, squeeze_145, buf397, primals_146, buf400, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_48
        del primals_146
        del squeeze_145
        del unsqueeze_1078
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf401 = aten.convolution_backward(buf400, relu_44, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_145
        buf402 = buf401[0]
        buf403 = buf401[1]
        del buf401
        buf404 = buf390; del buf390  # reuse
        buf405 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf406 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_44, buf402, convolution_47, unsqueeze_1090, squeeze_142, buf404, buf405, buf406, 2048, 1568, grid=grid(2048), stream=stream0)
        buf407 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf407, relu_44, convolution_47, unsqueeze_1090, buf405, squeeze_142, buf404, primals_143, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_47
        del primals_143
        del relu_44
        del squeeze_142
        del unsqueeze_1090
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf408 = aten.convolution_backward(buf407, relu_43, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf407
        del primals_142
        buf409 = buf408[0]
        buf410 = buf408[1]
        del buf408
        buf411 = buf405; del buf405  # reuse
        buf412 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf413 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_43, buf409, convolution_46, unsqueeze_1102, squeeze_139, buf411, buf412, buf413, 2048, 1568, grid=grid(2048), stream=stream0)
        buf414 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf414, relu_43, convolution_46, unsqueeze_1102, buf412, squeeze_139, buf411, primals_140, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_46
        del primals_140
        del relu_43
        del squeeze_139
        del unsqueeze_1102
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf415 = aten.convolution_backward(buf414, relu_42, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf414
        del primals_139
        buf416 = buf415[0]
        buf417 = buf415[1]
        del buf415
        buf418 = buf398; del buf398  # reuse
        buf419 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf421 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_42, buf396, buf416, convolution_45, unsqueeze_1114, squeeze_136, buf418, buf419, buf421, 1024, 1568, grid=grid(1024), stream=stream0)
        buf420 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_42, buf396, buf416, convolution_45, unsqueeze_1114, buf419, squeeze_136, buf418, primals_137, buf420, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_45
        del primals_137
        del squeeze_136
        del unsqueeze_1114
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf422 = aten.convolution_backward(buf420, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf420
        del primals_136
        buf423 = buf422[0]
        buf424 = buf422[1]
        del buf422
        buf425 = buf412; del buf412  # reuse
        buf426 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf427 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_41, buf423, convolution_44, unsqueeze_1126, squeeze_133, buf425, buf426, buf427, 2048, 1568, grid=grid(2048), stream=stream0)
        buf428 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf428, relu_41, convolution_44, unsqueeze_1126, buf426, squeeze_133, buf425, primals_134, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_44
        del primals_134
        del relu_41
        del squeeze_133
        del unsqueeze_1126
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf429 = aten.convolution_backward(buf428, relu_40, primals_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf428
        del primals_133
        buf430 = buf429[0]
        buf431 = buf429[1]
        del buf429
        buf432 = buf426; del buf426  # reuse
        buf433 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf434 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_40, buf430, convolution_43, unsqueeze_1138, squeeze_130, buf432, buf433, buf434, 2048, 1568, grid=grid(2048), stream=stream0)
        buf435 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf435, relu_40, convolution_43, unsqueeze_1138, buf433, squeeze_130, buf432, primals_131, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_43
        del primals_131
        del relu_40
        del squeeze_130
        del unsqueeze_1138
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf436 = aten.convolution_backward(buf435, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf435
        del primals_130
        buf437 = buf436[0]
        buf438 = buf436[1]
        del buf436
        buf439 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf439, relu_39, relu_42, buf416, buf437, 1605632, grid=grid(1605632), stream=stream0)
        del buf416
        del relu_39
        del relu_42
        buf440 = buf419; del buf419  # reuse
        buf441 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf442 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf439, convolution_42, unsqueeze_1150, squeeze_127, buf440, buf441, buf442, 1024, 1568, grid=grid(1024), stream=stream0)
        buf443 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf439, convolution_42, unsqueeze_1150, buf441, squeeze_127, buf440, primals_128, buf443, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_42
        del primals_128
        del squeeze_127
        del unsqueeze_1150
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf444 = aten.convolution_backward(buf443, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_127
        buf445 = buf444[0]
        buf446 = buf444[1]
        del buf444
        buf447 = buf433; del buf433  # reuse
        buf448 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf449 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_38, buf445, convolution_41, unsqueeze_1162, squeeze_124, buf447, buf448, buf449, 2048, 1568, grid=grid(2048), stream=stream0)
        buf450 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf450, relu_38, convolution_41, unsqueeze_1162, buf448, squeeze_124, buf447, primals_125, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_41
        del primals_125
        del relu_38
        del squeeze_124
        del unsqueeze_1162
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf451 = aten.convolution_backward(buf450, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf450
        del primals_124
        buf452 = buf451[0]
        buf453 = buf451[1]
        del buf451
        buf454 = buf448; del buf448  # reuse
        buf455 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf456 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_37, buf452, convolution_40, unsqueeze_1174, squeeze_121, buf454, buf455, buf456, 2048, 1568, grid=grid(2048), stream=stream0)
        buf457 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf457, relu_37, convolution_40, unsqueeze_1174, buf455, squeeze_121, buf454, primals_122, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_40
        del primals_122
        del relu_37
        del squeeze_121
        del unsqueeze_1174
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf458 = aten.convolution_backward(buf457, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf457
        del primals_121
        buf459 = buf458[0]
        buf460 = buf458[1]
        del buf458
        buf461 = buf441; del buf441  # reuse
        buf462 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf464 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_36, buf439, buf459, convolution_39, unsqueeze_1186, squeeze_118, buf461, buf462, buf464, 1024, 1568, grid=grid(1024), stream=stream0)
        buf463 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_36, buf439, buf459, convolution_39, unsqueeze_1186, buf462, squeeze_118, buf461, primals_119, buf463, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_39
        del primals_119
        del squeeze_118
        del unsqueeze_1186
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf465 = aten.convolution_backward(buf463, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf463
        del primals_118
        buf466 = buf465[0]
        buf467 = buf465[1]
        del buf465
        buf468 = buf455; del buf455  # reuse
        buf469 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf470 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_35, buf466, convolution_38, unsqueeze_1198, squeeze_115, buf468, buf469, buf470, 2048, 1568, grid=grid(2048), stream=stream0)
        buf471 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf471, relu_35, convolution_38, unsqueeze_1198, buf469, squeeze_115, buf468, primals_116, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_38
        del primals_116
        del relu_35
        del squeeze_115
        del unsqueeze_1198
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf472 = aten.convolution_backward(buf471, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf471
        del primals_115
        buf473 = buf472[0]
        buf474 = buf472[1]
        del buf472
        buf475 = buf469; del buf469  # reuse
        buf476 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf477 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_34, buf473, convolution_37, unsqueeze_1210, squeeze_112, buf475, buf476, buf477, 2048, 1568, grid=grid(2048), stream=stream0)
        buf478 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf478, relu_34, convolution_37, unsqueeze_1210, buf476, squeeze_112, buf475, primals_113, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_37
        del primals_113
        del relu_34
        del squeeze_112
        del unsqueeze_1210
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf479 = aten.convolution_backward(buf478, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf478
        del primals_112
        buf480 = buf479[0]
        buf481 = buf479[1]
        del buf479
        buf482 = buf439; del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf482, relu_33, relu_36, buf459, buf480, 1605632, grid=grid(1605632), stream=stream0)
        del buf459
        del relu_33
        del relu_36
        buf483 = buf462; del buf462  # reuse
        buf484 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf485 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf482, convolution_36, unsqueeze_1222, squeeze_109, buf483, buf484, buf485, 1024, 1568, grid=grid(1024), stream=stream0)
        buf486 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf482, convolution_36, unsqueeze_1222, buf484, squeeze_109, buf483, primals_110, buf486, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_36
        del primals_110
        del squeeze_109
        del unsqueeze_1222
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf487 = aten.convolution_backward(buf486, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_109
        buf488 = buf487[0]
        buf489 = buf487[1]
        del buf487
        buf490 = buf476; del buf476  # reuse
        buf491 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf492 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_32, buf488, convolution_35, unsqueeze_1234, squeeze_106, buf490, buf491, buf492, 2048, 1568, grid=grid(2048), stream=stream0)
        buf493 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf493, relu_32, convolution_35, unsqueeze_1234, buf491, squeeze_106, buf490, primals_107, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_35
        del primals_107
        del relu_32
        del squeeze_106
        del unsqueeze_1234
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf494 = aten.convolution_backward(buf493, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf493
        del primals_106
        buf495 = buf494[0]
        buf496 = buf494[1]
        del buf494
        buf497 = buf491; del buf491  # reuse
        buf498 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf499 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_31, buf495, convolution_34, unsqueeze_1246, squeeze_103, buf497, buf498, buf499, 2048, 1568, grid=grid(2048), stream=stream0)
        buf500 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf500, relu_31, convolution_34, unsqueeze_1246, buf498, squeeze_103, buf497, primals_104, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_34
        del primals_104
        del relu_31
        del squeeze_103
        del unsqueeze_1246
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf501 = aten.convolution_backward(buf500, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf500
        del primals_103
        buf502 = buf501[0]
        buf503 = buf501[1]
        del buf501
        buf504 = buf484; del buf484  # reuse
        buf505 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf507 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_30, buf482, buf502, convolution_33, unsqueeze_1258, squeeze_100, buf504, buf505, buf507, 1024, 1568, grid=grid(1024), stream=stream0)
        buf506 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_30, buf482, buf502, convolution_33, unsqueeze_1258, buf505, squeeze_100, buf504, primals_101, buf506, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_33
        del primals_101
        del squeeze_100
        del unsqueeze_1258
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf508 = aten.convolution_backward(buf506, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf506
        del primals_100
        buf509 = buf508[0]
        buf510 = buf508[1]
        del buf508
        buf511 = buf498; del buf498  # reuse
        buf512 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf513 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_29, buf509, convolution_32, unsqueeze_1270, squeeze_97, buf511, buf512, buf513, 2048, 1568, grid=grid(2048), stream=stream0)
        buf514 = buf509; del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf514, relu_29, convolution_32, unsqueeze_1270, buf512, squeeze_97, buf511, primals_98, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_32
        del primals_98
        del relu_29
        del squeeze_97
        del unsqueeze_1270
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf515 = aten.convolution_backward(buf514, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf514
        del primals_97
        buf516 = buf515[0]
        buf517 = buf515[1]
        del buf515
        buf518 = buf512; del buf512  # reuse
        buf519 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf520 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_28, buf516, convolution_31, unsqueeze_1282, squeeze_94, buf518, buf519, buf520, 2048, 1568, grid=grid(2048), stream=stream0)
        buf521 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf521, relu_28, convolution_31, unsqueeze_1282, buf519, squeeze_94, buf518, primals_95, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_31
        del primals_95
        del relu_28
        del squeeze_94
        del unsqueeze_1282
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf522 = aten.convolution_backward(buf521, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf521
        del primals_94
        buf523 = buf522[0]
        buf524 = buf522[1]
        del buf522
        buf525 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_16.run(buf525, relu_27, relu_30, buf502, buf523, 1605632, grid=grid(1605632), stream=stream0)
        del relu_27
        del relu_30
        buf526 = buf505; del buf505  # reuse
        buf527 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf528 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf525, convolution_30, unsqueeze_1294, squeeze_91, buf526, buf527, buf528, 1024, 1568, grid=grid(1024), stream=stream0)
        buf529 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf525, convolution_30, unsqueeze_1294, buf527, squeeze_91, buf526, primals_92, buf529, 1605632, grid=grid(1605632), stream=stream0)
        del convolution_30
        del primals_92
        del squeeze_91
        del unsqueeze_1294
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf530 = aten.convolution_backward(buf529, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_91
        buf531 = buf530[0]
        buf532 = buf530[1]
        del buf530
        buf533 = buf519; del buf519  # reuse
        buf534 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf535 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_26, buf531, convolution_29, unsqueeze_1306, squeeze_88, buf533, buf534, buf535, 2048, 1568, grid=grid(2048), stream=stream0)
        buf536 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf536, relu_26, convolution_29, unsqueeze_1306, buf534, squeeze_88, buf533, primals_89, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_29
        del primals_89
        del relu_26
        del squeeze_88
        del unsqueeze_1306
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf537 = aten.convolution_backward(buf536, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf536
        del primals_88
        buf538 = buf537[0]
        buf539 = buf537[1]
        del buf537
        buf540 = buf534; del buf534  # reuse
        buf541 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf542 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_25, buf538, convolution_28, unsqueeze_1318, squeeze_85, buf540, buf541, buf542, 2048, 1568, grid=grid(2048), stream=stream0)
        buf543 = buf538; del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf543, relu_25, convolution_28, unsqueeze_1318, buf541, squeeze_85, buf540, primals_86, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_28
        del primals_86
        del relu_25
        del squeeze_85
        del unsqueeze_1318
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf544 = aten.convolution_backward(buf543, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf543
        del primals_85
        buf545 = buf544[0]
        buf546 = buf544[1]
        del buf544
        buf547 = buf527; del buf527  # reuse
        buf548 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf554 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf550 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf556 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(relu_24, buf525, buf545, convolution_27, unsqueeze_1330, convolution_26, unsqueeze_1342, squeeze_82, squeeze_79, buf547, buf548, buf554, buf550, buf556, 1024, 1568, grid=grid(1024), stream=stream0)
        buf549 = buf529; del buf529  # reuse
        buf555 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_24, buf525, buf545, convolution_27, unsqueeze_1330, buf548, squeeze_82, buf547, primals_83, convolution_26, unsqueeze_1342, buf554, squeeze_79, primals_80, buf549, buf555, 1605632, grid=grid(1605632), stream=stream0)
        del buf525
        del buf545
        del convolution_26
        del convolution_27
        del primals_80
        del primals_83
        del relu_24
        del squeeze_79
        del squeeze_82
        del unsqueeze_1330
        del unsqueeze_1342
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf551 = aten.convolution_backward(buf549, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf549
        del primals_82
        buf552 = buf551[0]
        buf553 = buf551[1]
        del buf551
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf557 = aten.convolution_backward(buf555, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf555
        del primals_79
        buf558 = buf557[0]
        buf559 = buf557[1]
        del buf557
        buf560 = buf541; del buf541  # reuse
        buf561 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf562 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(relu_23, buf558, convolution_25, unsqueeze_1354, squeeze_76, buf560, buf561, buf562, 2048, 1568, grid=grid(2048), stream=stream0)
        buf563 = buf558; del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15.run(buf563, relu_23, convolution_25, unsqueeze_1354, buf561, squeeze_76, buf560, primals_77, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_25
        del primals_77
        del relu_23
        del squeeze_76
        del unsqueeze_1354
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf564 = aten.convolution_backward(buf563, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_76
        buf565 = buf564[0]
        buf566 = buf564[1]
        del buf564
        buf567 = buf561; del buf561  # reuse
        buf568 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf569 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_22, buf565, convolution_24, unsqueeze_1366, squeeze_73, buf567, buf568, buf569, 2048, 6272, grid=grid(2048), stream=stream0)
        buf570 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23.run(buf570, relu_22, convolution_24, unsqueeze_1366, buf568, squeeze_73, buf567, primals_74, 12845056, grid=grid(12845056), stream=stream0)
        del buf568
        del convolution_24
        del primals_74
        del relu_22
        del squeeze_73
        del unsqueeze_1366
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf571 = aten.convolution_backward(buf570, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf570
        del primals_73
        buf572 = buf571[0]
        buf573 = buf571[1]
        del buf571
        buf574 = empty((512, ), device='cuda', dtype=torch.float32)
        buf575 = empty((512, ), device='cuda', dtype=torch.float32)
        buf577 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_24.run(relu_21, buf552, buf572, convolution_23, unsqueeze_1378, squeeze_70, buf574, buf575, buf577, 512, 6272, grid=grid(512), stream=stream0)
        buf576 = reinterpret_tensor(buf563, (8, 512, 28, 28), (401408, 784, 28, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_25.run(relu_21, buf552, buf572, convolution_23, unsqueeze_1378, buf575, squeeze_70, buf574, primals_71, buf576, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_23
        del primals_71
        del squeeze_70
        del unsqueeze_1378
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf578 = aten.convolution_backward(buf576, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf576
        del primals_70
        buf579 = buf578[0]
        buf580 = buf578[1]
        del buf578
        buf581 = buf554; del buf554  # reuse
        buf582 = buf548; del buf548  # reuse
        buf583 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_20, buf579, convolution_22, unsqueeze_1390, squeeze_67, buf581, buf582, buf583, 1024, 6272, grid=grid(1024), stream=stream0)
        buf584 = buf579; del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(buf584, relu_20, convolution_22, unsqueeze_1390, buf582, squeeze_67, buf581, primals_68, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_22
        del primals_68
        del relu_20
        del squeeze_67
        del unsqueeze_1390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf585 = aten.convolution_backward(buf584, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf584
        del primals_67
        buf586 = buf585[0]
        buf587 = buf585[1]
        del buf585
        buf588 = buf582; del buf582  # reuse
        buf589 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf590 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_19, buf586, convolution_21, unsqueeze_1402, squeeze_64, buf588, buf589, buf590, 1024, 6272, grid=grid(1024), stream=stream0)
        buf591 = buf586; del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(buf591, relu_19, convolution_21, unsqueeze_1402, buf589, squeeze_64, buf588, primals_65, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_21
        del primals_65
        del relu_19
        del squeeze_64
        del unsqueeze_1402
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf592 = aten.convolution_backward(buf591, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf591
        del primals_64
        buf593 = buf592[0]
        buf594 = buf592[1]
        del buf592
        buf595 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_28.run(buf595, relu_18, relu_21, buf572, buf593, 3211264, grid=grid(3211264), stream=stream0)
        del buf572
        del relu_18
        del relu_21
        buf596 = buf575; del buf575  # reuse
        buf597 = empty((512, ), device='cuda', dtype=torch.float32)
        buf598 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_29.run(buf595, convolution_20, unsqueeze_1414, squeeze_61, buf596, buf597, buf598, 512, 6272, grid=grid(512), stream=stream0)
        buf599 = buf593; del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_30.run(buf595, convolution_20, unsqueeze_1414, buf597, squeeze_61, buf596, primals_62, buf599, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_20
        del primals_62
        del squeeze_61
        del unsqueeze_1414
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf600 = aten.convolution_backward(buf599, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_61
        buf601 = buf600[0]
        buf602 = buf600[1]
        del buf600
        buf603 = buf589; del buf589  # reuse
        buf604 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf605 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_17, buf601, convolution_19, unsqueeze_1426, squeeze_58, buf603, buf604, buf605, 1024, 6272, grid=grid(1024), stream=stream0)
        buf606 = buf601; del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(buf606, relu_17, convolution_19, unsqueeze_1426, buf604, squeeze_58, buf603, primals_59, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_19
        del primals_59
        del relu_17
        del squeeze_58
        del unsqueeze_1426
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf607 = aten.convolution_backward(buf606, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf606
        del primals_58
        buf608 = buf607[0]
        buf609 = buf607[1]
        del buf607
        buf610 = buf604; del buf604  # reuse
        buf611 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf612 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_16, buf608, convolution_18, unsqueeze_1438, squeeze_55, buf610, buf611, buf612, 1024, 6272, grid=grid(1024), stream=stream0)
        buf613 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(buf613, relu_16, convolution_18, unsqueeze_1438, buf611, squeeze_55, buf610, primals_56, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_18
        del primals_56
        del relu_16
        del squeeze_55
        del unsqueeze_1438
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf614 = aten.convolution_backward(buf613, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf613
        del primals_55
        buf615 = buf614[0]
        buf616 = buf614[1]
        del buf614
        buf617 = buf597; del buf597  # reuse
        buf618 = empty((512, ), device='cuda', dtype=torch.float32)
        buf620 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_24.run(relu_15, buf595, buf615, convolution_17, unsqueeze_1450, squeeze_52, buf617, buf618, buf620, 512, 6272, grid=grid(512), stream=stream0)
        buf619 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_25.run(relu_15, buf595, buf615, convolution_17, unsqueeze_1450, buf618, squeeze_52, buf617, primals_53, buf619, 3211264, grid=grid(3211264), stream=stream0)
        del convolution_17
        del primals_53
        del squeeze_52
        del unsqueeze_1450
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf621 = aten.convolution_backward(buf619, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf619
        del primals_52
        buf622 = buf621[0]
        buf623 = buf621[1]
        del buf621
        buf624 = buf611; del buf611  # reuse
        buf625 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf626 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_14, buf622, convolution_16, unsqueeze_1462, squeeze_49, buf624, buf625, buf626, 1024, 6272, grid=grid(1024), stream=stream0)
        buf627 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(buf627, relu_14, convolution_16, unsqueeze_1462, buf625, squeeze_49, buf624, primals_50, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_16
        del primals_50
        del relu_14
        del squeeze_49
        del unsqueeze_1462
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf628 = aten.convolution_backward(buf627, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf627
        del primals_49
        buf629 = buf628[0]
        buf630 = buf628[1]
        del buf628
        buf631 = buf625; del buf625  # reuse
        buf632 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf633 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_13, buf629, convolution_15, unsqueeze_1474, squeeze_46, buf631, buf632, buf633, 1024, 6272, grid=grid(1024), stream=stream0)
        buf634 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(buf634, relu_13, convolution_15, unsqueeze_1474, buf632, squeeze_46, buf631, primals_47, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_15
        del primals_47
        del relu_13
        del squeeze_46
        del unsqueeze_1474
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf635 = aten.convolution_backward(buf634, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf634
        del primals_46
        buf636 = buf635[0]
        buf637 = buf635[1]
        del buf635
        buf638 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_28.run(buf638, relu_12, relu_15, buf615, buf636, 3211264, grid=grid(3211264), stream=stream0)
        del relu_12
        del relu_15
        buf639 = buf618; del buf618  # reuse
        buf640 = empty((512, ), device='cuda', dtype=torch.float32)
        buf646 = empty((512, ), device='cuda', dtype=torch.float32)
        buf641 = empty((512, ), device='cuda', dtype=torch.float32)
        buf647 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf638, convolution_14, unsqueeze_1486, convolution_13, unsqueeze_1498, squeeze_43, squeeze_40, buf639, buf640, buf646, buf641, buf647, 512, 6272, grid=grid(512), stream=stream0)
        buf642 = buf636; del buf636  # reuse
        buf648 = buf615; del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_32.run(buf638, convolution_14, unsqueeze_1486, buf640, squeeze_43, buf639, primals_44, convolution_13, unsqueeze_1498, buf646, squeeze_40, primals_41, buf642, buf648, 3211264, grid=grid(3211264), stream=stream0)
        del buf638
        del convolution_13
        del convolution_14
        del primals_41
        del primals_44
        del squeeze_40
        del squeeze_43
        del unsqueeze_1486
        del unsqueeze_1498
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf643 = aten.convolution_backward(buf642, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf642
        del primals_43
        buf644 = buf643[0]
        buf645 = buf643[1]
        del buf643
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf649 = aten.convolution_backward(buf648, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf648
        del primals_40
        buf650 = buf649[0]
        buf651 = buf649[1]
        del buf649
        buf652 = buf632; del buf632  # reuse
        buf653 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf654 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_11, buf650, convolution_12, unsqueeze_1510, squeeze_37, buf652, buf653, buf654, 1024, 6272, grid=grid(1024), stream=stream0)
        buf655 = buf650; del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(buf655, relu_11, convolution_12, unsqueeze_1510, buf653, squeeze_37, buf652, primals_38, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_12
        del primals_38
        del relu_11
        del squeeze_37
        del unsqueeze_1510
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf656 = aten.convolution_backward(buf655, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_37
        buf657 = buf656[0]
        buf658 = buf656[1]
        del buf656
        buf659 = buf653; del buf653  # reuse
        buf660 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf661 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_33.run(relu_10, buf657, convolution_11, unsqueeze_1522, squeeze_34, buf659, buf660, buf661, 1024, 25088, grid=grid(1024), stream=stream0)
        buf662 = buf657; del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34.run(buf662, relu_10, convolution_11, unsqueeze_1522, buf660, squeeze_34, buf659, primals_35, 25690112, grid=grid(25690112), stream=stream0)
        del buf660
        del convolution_11
        del primals_35
        del relu_10
        del squeeze_34
        del unsqueeze_1522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf663 = aten.convolution_backward(buf662, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf662
        del primals_34
        buf664 = buf663[0]
        buf665 = buf663[1]
        del buf663
        buf666 = empty((256, ), device='cuda', dtype=torch.float32)
        buf667 = empty((256, ), device='cuda', dtype=torch.float32)
        buf669 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_9, buf644, buf664, convolution_10, unsqueeze_1534, squeeze_31, buf666, buf667, buf669, 256, 25088, grid=grid(256), stream=stream0)
        buf668 = reinterpret_tensor(buf655, (8, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_9, buf644, buf664, convolution_10, unsqueeze_1534, buf667, squeeze_31, buf666, primals_32, buf668, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_10
        del primals_32
        del squeeze_31
        del unsqueeze_1534
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf670 = aten.convolution_backward(buf668, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf668
        del primals_31
        buf671 = buf670[0]
        buf672 = buf670[1]
        del buf670
        buf673 = buf646; del buf646  # reuse
        buf674 = buf640; del buf640  # reuse
        buf675 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_8, buf671, convolution_9, unsqueeze_1546, squeeze_28, buf673, buf674, buf675, 512, 25088, grid=grid(512), stream=stream0)
        buf676 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf676, relu_8, convolution_9, unsqueeze_1546, buf674, squeeze_28, buf673, primals_29, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_9
        del primals_29
        del relu_8
        del squeeze_28
        del unsqueeze_1546
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf677 = aten.convolution_backward(buf676, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf676
        del primals_28
        buf678 = buf677[0]
        buf679 = buf677[1]
        del buf677
        buf680 = buf674; del buf674  # reuse
        buf681 = empty((512, ), device='cuda', dtype=torch.float32)
        buf682 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_7, buf678, convolution_8, unsqueeze_1558, squeeze_25, buf680, buf681, buf682, 512, 25088, grid=grid(512), stream=stream0)
        buf683 = buf678; del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf683, relu_7, convolution_8, unsqueeze_1558, buf681, squeeze_25, buf680, primals_26, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_8
        del primals_26
        del relu_7
        del squeeze_25
        del unsqueeze_1558
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf684 = aten.convolution_backward(buf683, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf683
        del primals_25
        buf685 = buf684[0]
        buf686 = buf684[1]
        del buf684
        buf687 = buf644; del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_39.run(buf687, relu_6, relu_9, buf664, buf685, 6422528, grid=grid(6422528), stream=stream0)
        del relu_6
        del relu_9
        buf688 = buf667; del buf667  # reuse
        buf689 = empty((256, ), device='cuda', dtype=torch.float32)
        buf690 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_40.run(buf687, convolution_7, unsqueeze_1570, squeeze_22, buf688, buf689, buf690, 256, 25088, grid=grid(256), stream=stream0)
        buf691 = buf685; del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_41.run(buf687, convolution_7, unsqueeze_1570, buf689, squeeze_22, buf688, primals_23, buf691, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_7
        del primals_23
        del squeeze_22
        del unsqueeze_1570
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf692 = aten.convolution_backward(buf691, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        buf693 = buf692[0]
        buf694 = buf692[1]
        del buf692
        buf695 = buf681; del buf681  # reuse
        buf696 = empty((512, ), device='cuda', dtype=torch.float32)
        buf697 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_5, buf693, convolution_6, unsqueeze_1582, squeeze_19, buf695, buf696, buf697, 512, 25088, grid=grid(512), stream=stream0)
        buf698 = buf693; del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf698, relu_5, convolution_6, unsqueeze_1582, buf696, squeeze_19, buf695, primals_20, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_6
        del primals_20
        del relu_5
        del squeeze_19
        del unsqueeze_1582
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf699 = aten.convolution_backward(buf698, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf698
        del primals_19
        buf700 = buf699[0]
        buf701 = buf699[1]
        del buf699
        buf702 = buf696; del buf696  # reuse
        buf703 = empty((512, ), device='cuda', dtype=torch.float32)
        buf704 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_4, buf700, convolution_5, unsqueeze_1594, squeeze_16, buf702, buf703, buf704, 512, 25088, grid=grid(512), stream=stream0)
        buf705 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf705, relu_4, convolution_5, unsqueeze_1594, buf703, squeeze_16, buf702, primals_17, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_5
        del primals_17
        del relu_4
        del squeeze_16
        del unsqueeze_1594
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf706 = aten.convolution_backward(buf705, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf705
        del primals_16
        buf707 = buf706[0]
        buf708 = buf706[1]
        del buf706
        buf709 = buf689; del buf689  # reuse
        buf710 = empty((256, ), device='cuda', dtype=torch.float32)
        buf716 = empty((256, ), device='cuda', dtype=torch.float32)
        buf712 = empty((256, ), device='cuda', dtype=torch.float32)
        buf718 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_42.run(relu_3, buf687, buf707, convolution_4, unsqueeze_1606, convolution_3, unsqueeze_1618, squeeze_13, squeeze_10, buf709, buf710, buf716, buf712, buf718, 256, 25088, grid=grid(256), stream=stream0)
        buf711 = buf691; del buf691  # reuse
        buf717 = buf664; del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_43.run(relu_3, buf687, buf707, convolution_4, unsqueeze_1606, buf710, squeeze_13, buf709, primals_14, convolution_3, unsqueeze_1618, buf716, squeeze_10, primals_11, buf711, buf717, 6422528, grid=grid(6422528), stream=stream0)
        del buf687
        del buf707
        del buf710
        del buf716
        del convolution_3
        del convolution_4
        del primals_11
        del primals_14
        del relu_3
        del squeeze_10
        del squeeze_13
        del unsqueeze_1606
        del unsqueeze_1618
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf713 = aten.convolution_backward(buf711, getitem_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf711
        del primals_13
        buf714 = buf713[0]
        buf715 = buf713[1]
        del buf713
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf719 = aten.convolution_backward(buf717, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_10
        buf720 = buf719[0]
        buf721 = buf719[1]
        del buf719
        buf722 = buf703; del buf703  # reuse
        buf723 = empty((512, ), device='cuda', dtype=torch.float32)
        buf724 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_2, buf720, convolution_2, unsqueeze_1630, squeeze_7, buf722, buf723, buf724, 512, 25088, grid=grid(512), stream=stream0)
        buf725 = buf720; del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf725, relu_2, convolution_2, unsqueeze_1630, buf723, squeeze_7, buf722, primals_8, 12845056, grid=grid(12845056), stream=stream0)
        del convolution_2
        del primals_8
        del relu_2
        del squeeze_7
        del unsqueeze_1630
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf726 = aten.convolution_backward(buf725, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf725
        del primals_7
        buf727 = buf726[0]
        buf728 = buf726[1]
        del buf726
        buf729 = buf723; del buf723  # reuse
        buf730 = empty((512, ), device='cuda', dtype=torch.float32)
        buf731 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_1, buf727, convolution_1, unsqueeze_1642, squeeze_4, buf729, buf730, buf731, 512, 25088, grid=grid(512), stream=stream0)
        buf732 = buf727; del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf732, relu_1, convolution_1, unsqueeze_1642, buf730, squeeze_4, buf729, primals_5, 12845056, grid=grid(12845056), stream=stream0)
        del buf730
        del convolution_1
        del primals_5
        del relu_1
        del squeeze_4
        del unsqueeze_1642
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf733 = aten.convolution_backward(buf732, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf732
        del getitem_2
        del primals_4
        buf734 = buf733[0]
        buf735 = buf733[1]
        del buf733
        buf736 = buf714; del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf736, buf734, 1605632, grid=grid(1605632), stream=stream0)
        del buf734
        buf737 = reinterpret_tensor(buf717, (8, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        triton_poi_fused_add_max_pool2d_with_indices_backward_45.run(getitem_3, buf736, buf737, 6422528, grid=grid(6422528), stream=stream0)
        del buf736
        del getitem_3
        buf738 = empty((64, 13), device='cuda', dtype=torch.float32)
        buf740 = empty((64, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu, buf737, convolution, unsqueeze_1654, buf738, buf740, 832, 7720, grid=grid(832), stream=stream0)
        buf739 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf738, buf739, 64, 13, grid=grid(64), stream=stream0)
        del buf738
        buf741 = empty((64, ), device='cuda', dtype=torch.float32)
        buf742 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf740, squeeze_1, buf741, buf742, 64, 13, grid=grid(64), stream=stream0)
        del buf740
        buf743 = buf737; del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(buf743, relu, convolution, unsqueeze_1654, buf741, squeeze_1, buf739, primals_2, 6422528, grid=grid(6422528), stream=stream0)
        del buf741
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_1654
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf744 = aten.convolution_backward(buf743, primals_627, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf743
        del primals_1
        del primals_627
        buf745 = buf744[1]
        return (buf745, buf742, buf739, buf735, buf731, buf729, buf728, buf724, buf722, buf721, buf718, buf709, buf715, buf712, buf709, buf708, buf704, buf702, buf701, buf697, buf695, buf694, buf690, buf688, buf686, buf682, buf680, buf679, buf675, buf673, buf672, buf669, buf666, buf665, buf661, buf659, buf658, buf654, buf652, buf651, buf647, buf639, buf645, buf641, buf639, buf637, buf633, buf631, buf630, buf626, buf624, buf623, buf620, buf617, buf616, buf612, buf610, buf609, buf605, buf603, buf602, buf598, buf596, buf594, buf590, buf588, buf587, buf583, buf581, buf580, buf577, buf574, buf573, buf569, buf567, buf566, buf562, buf560, buf559, buf556, buf547, buf553, buf550, buf547, buf546, buf542, buf540, buf539, buf535, buf533, buf532, buf528, buf526, buf524, buf520, buf518, buf517, buf513, buf511, buf510, buf507, buf504, buf503, buf499, buf497, buf496, buf492, buf490, buf489, buf485, buf483, buf481, buf477, buf475, buf474, buf470, buf468, buf467, buf464, buf461, buf460, buf456, buf454, buf453, buf449, buf447, buf446, buf442, buf440, buf438, buf434, buf432, buf431, buf427, buf425, buf424, buf421, buf418, buf417, buf413, buf411, buf410, buf406, buf404, buf403, buf399, buf397, buf395, buf391, buf389, buf388, buf384, buf382, buf381, buf378, buf375, buf374, buf370, buf368, buf367, buf363, buf361, buf360, buf356, buf354, buf352, buf348, buf346, buf345, buf341, buf339, buf338, buf335, buf332, buf331, buf327, buf325, buf324, buf320, buf318, buf317, buf313, buf311, buf309, buf305, buf303, buf302, buf298, buf296, buf295, buf292, buf289, buf288, buf284, buf282, buf281, buf277, buf275, buf274, buf270, buf268, buf266, buf262, buf260, buf259, buf255, buf253, buf252, buf249, buf246, buf245, buf241, buf239, buf238, buf234, buf232, buf231, buf227, buf225, buf223, buf219, buf217, buf216, buf212, buf210, buf209, buf206, buf203, buf202, buf198, buf196, buf195, buf191, buf189, buf188, buf184, buf182, buf180, buf176, buf174, buf173, buf169, buf167, buf166, buf163, buf160, buf159, buf155, buf153, buf152, buf148, buf146, buf145, buf141, buf139, buf137, buf133, buf131, buf130, buf126, buf124, buf123, buf120, buf117, buf116, buf112, buf110, buf109, buf105, buf103, buf102, buf98, buf96, buf94, buf90, buf88, buf87, buf83, buf81, buf80, buf77, buf74, buf73, buf69, buf67, buf66, buf62, buf60, buf59, buf55, buf47, buf53, buf49, buf47, buf45, buf41, buf39, buf38, buf34, buf32, buf31, buf27, buf24, buf23, buf19, buf17, buf16, buf12, buf10, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 64, 56, 56), (200704, 3136, 56, 1), device='cuda:0', dtype=torch.int64)
    convolution_1 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 512, 56, 56), (1605632, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 256, 56, 56), (802816, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 1024, 56, 56), (3211264, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 1024, 56, 56), (3211264, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 1024, 28, 28), (802816, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((8, 512, 28, 28), (401408, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 2048, 28, 28), (1605632, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 2048, 28, 28), (1605632, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_48 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_49 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_52 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_53 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_54 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_56 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_57 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_58 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_59 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_60 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_61 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_62 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_63 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_64 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_205 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_66 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_67 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_68 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_217 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_69 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_71 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_72 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_229 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_73 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_74 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_76 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_241 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_77 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_78 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_79 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_80 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_253 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_81 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_256 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_82 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_259 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_83 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_262 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_84 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_265 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_85 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_268 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_86 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_90 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_87 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_274 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_88 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_277 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_89 = rand_strided((8, 2048, 14, 14), (401408, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_90 = rand_strided((8, 1024, 14, 14), (200704, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_94 = rand_strided((8, 4096, 14, 14), (802816, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_283 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_91 = rand_strided((8, 4096, 14, 14), (802816, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_286 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_92 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_96 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_289 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_97 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_292 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_93 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_98 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_295 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_94 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_99 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_298 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_95 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_100 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_301 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_96 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_101 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_304 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_97 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_102 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_307 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_98 = rand_strided((8, 4096, 7, 7), (200704, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_103 = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_310 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 2048, 7, 7), (100352, 49, 7, 1), device='cuda:0', dtype=torch.bool)
    unsqueeze_418 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_706 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_754 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_802 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_850 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_862 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_874 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_886 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_898 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_910 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_922 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_934 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_946 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_958 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_970 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_982 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_994 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1006 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1018 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1030 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1042 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1054 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1066 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1078 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1090 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1102 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1114 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1126 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1138 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1150 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1162 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1174 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1186 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1198 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1210 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1222 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1234 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1246 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1258 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1270 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1282 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1294 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1306 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1318 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1330 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1342 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1354 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1366 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1378 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1390 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1402 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1414 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1426 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1438 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1450 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1462 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1474 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1486 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1498 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1510 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1522 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1534 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1546 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1558 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1570 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1582 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1594 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1606 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1618 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1630 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1642 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1654 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_627, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, relu_42, convolution_46, squeeze_139, relu_43, convolution_47, squeeze_142, relu_44, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, relu_46, convolution_50, squeeze_151, relu_47, convolution_51, squeeze_154, relu_48, convolution_52, squeeze_157, relu_49, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, relu_51, convolution_55, squeeze_166, relu_52, convolution_56, squeeze_169, relu_53, convolution_57, squeeze_172, relu_54, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, relu_56, convolution_60, squeeze_181, relu_57, convolution_61, squeeze_184, relu_58, convolution_62, squeeze_187, relu_59, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, relu_61, convolution_65, squeeze_196, relu_62, convolution_66, squeeze_199, relu_63, convolution_67, squeeze_202, relu_64, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, relu_66, convolution_70, squeeze_211, relu_67, convolution_71, squeeze_214, relu_68, convolution_72, squeeze_217, relu_69, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, relu_71, convolution_75, squeeze_226, relu_72, convolution_76, squeeze_229, relu_73, convolution_77, squeeze_232, relu_74, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, relu_76, convolution_80, squeeze_241, relu_77, convolution_81, squeeze_244, relu_78, convolution_82, squeeze_247, relu_79, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, relu_81, convolution_85, squeeze_256, relu_82, convolution_86, squeeze_259, relu_83, convolution_87, squeeze_262, relu_84, convolution_88, squeeze_265, relu_85, convolution_89, squeeze_268, relu_86, convolution_90, squeeze_271, relu_87, convolution_91, squeeze_274, relu_88, convolution_92, squeeze_277, relu_89, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, relu_91, convolution_95, squeeze_286, relu_92, convolution_96, squeeze_289, convolution_97, squeeze_292, relu_93, convolution_98, squeeze_295, relu_94, convolution_99, squeeze_298, relu_95, convolution_100, squeeze_301, relu_96, convolution_101, squeeze_304, relu_97, convolution_102, squeeze_307, relu_98, convolution_103, squeeze_310, view, permute_1, le, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, unsqueeze_1042, unsqueeze_1054, unsqueeze_1066, unsqueeze_1078, unsqueeze_1090, unsqueeze_1102, unsqueeze_1114, unsqueeze_1126, unsqueeze_1138, unsqueeze_1150, unsqueeze_1162, unsqueeze_1174, unsqueeze_1186, unsqueeze_1198, unsqueeze_1210, unsqueeze_1222, unsqueeze_1234, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, unsqueeze_1282, unsqueeze_1294, unsqueeze_1306, unsqueeze_1318, unsqueeze_1330, unsqueeze_1342, unsqueeze_1354, unsqueeze_1366, unsqueeze_1378, unsqueeze_1390, unsqueeze_1402, unsqueeze_1414, unsqueeze_1426, unsqueeze_1438, unsqueeze_1450, unsqueeze_1462, unsqueeze_1474, unsqueeze_1486, unsqueeze_1498, unsqueeze_1510, unsqueeze_1522, unsqueeze_1534, unsqueeze_1546, unsqueeze_1558, unsqueeze_1570, unsqueeze_1582, unsqueeze_1594, unsqueeze_1606, unsqueeze_1618, unsqueeze_1630, unsqueeze_1642, unsqueeze_1654, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swsl_resnext101_32x16d', benchmark_compiled_module)
