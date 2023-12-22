
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


# kernel path: /tmp/torchinductor_youkaichao/c7/cc7kxc2trnwgaitld2fjig7h2ntvqwrejljxa6ct5c5vfqrzubcw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1856 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 64.0
        tmp3 = tmp1 / tmp2
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


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dw3elecez4smpflnjqa2ctuznult5gjs6znsrrmteozwtuzcov.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/yg/cygagworirjbkeyufs4jorsngreqs42emras4fi5njy3kvusmqyx.py
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
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tkyfcxz6dt26fzzkhvly2c3anwbdd5wogahdvira5uvsf4lsa2.py
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
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 192
    x2 = (xindex // 12288)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (1856 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/rg/crgecuq5k74gmsqxspyhrqhx7aaipd2ra5ekw6do5ovsiahwsb7j.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 8
    x2 = (xindex // 64)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None)
    tmp25 = tl.load(in_ptr0 + ((8*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + ((8*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + ((8*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None)
    tmp42 = tl.load(in_ptr0 + ((8*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + ((8*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + ((8*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(8, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(8, 2 + x0)
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
    tl.store(out_ptr0 + (x4), tmp58, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvcm4vhnscrgluitjw7ij7v2nalokzvhhdy6kbfddbss6k5mje2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1472 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 64.0
        tmp3 = tmp1 / tmp2
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


# kernel path: /tmp/torchinductor_youkaichao/3i/c3imp2jdge7n2ypyzwnbkajejqzcwvqfx4mmrimsr7rdmlejlt6e.py
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
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zbsa6pkcl63szltzytmoartnivblr3yhiqd2khplbmhtxajcvn.py
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
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyy2fp7m5nanuwciksrqltq4wqa6wkzx6y5biuvmxaeykzre57g3.py
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 24576)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (1472 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/2l/c2l3smpu2yheutc55auchuzbyvf5au7atdbeouat2sgvhzoznfls.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1088 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 64.0
        tmp3 = tmp1 / tmp2
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


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyaifsrqdxly2hpagzw23xuak2s4567cegqxyehkn64dux7n25h.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 24576)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (1088 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/rq/crqicktexztlkuyaj7jdobuy4ty7awmwboio7ioea3ask6qcgmg6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 384
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 64
    r2 = (rindex // 64)
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (24576*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (64*x0) + (24576*r2)), rmask & xmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqk5bkb77bh5kihkl3q33h5k452dodcw6rz5gkdamgzhujtogvv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp8 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (24576*(r2 // 64)) + (49152*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((64*x0) + (24576*(r2 // 64)) + (49152*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cduv74skb3vavoyidqqzjwiqpms5xgnbqvnsu2dggfog47emfxf7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (24576*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (64*x2) + (24576*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.001953125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7f6yboojboanuzoeo7kczqkyl43kvbal5zhjxq6jcfktu5h5suv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1792
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 448
    x1 = (xindex // 448)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (448*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (28672*(r2 // 64)) + (57344*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (448*r2) + (57344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsu5brhub64ffn7ryyyqvuhlfcwgnob7zvw67xncq22tptb2ngz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxx5q2izazy32gb5fvnacn5suoih65wb5rgqsdik6ogabqwjxwr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (448*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfb5s2q3caavjtcj42ok5kks5brf7k3lsbbutnvq4rbcslokimus.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 448
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (28672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (448*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cwsrl5d6bnhylcssaqzvn5u3ktlqp6b57jylul6mskyrmzhx3qqj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (704 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 64.0
        tmp3 = tmp1 / tmp2
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


# kernel path: /tmp/torchinductor_youkaichao/6o/c6oluvtmfjysikgqpiwoiqpsj3hbbweovc42nlrxgczno3n43c76.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 24576)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (704 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/ez/ceznwx3nm3j2soecdop7yunlcglpbif5ae2eztx2feviogewhhis.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (320 + x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 64.0
        tmp3 = tmp1 / tmp2
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


# kernel path: /tmp/torchinductor_youkaichao/lx/clx4c2xuelt5hfyrfsnprmae5bnrpbghrnowo6rinnyhlz7vjrqm.py
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
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x2 = (xindex // 24576)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (320 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/2t/c2t7dmfu3fmwhkqvetxs5h4arlb5xoxya65xqdcctpn55nr7pjkr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (2048*(r2 // 64)) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 64.0
        tmp3 = tmp1 / tmp2
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5ly2wordnkwhlwqxeuuqxef3ckcgsk3x4ctp3m3qmvfdywlehx4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chtu3m5w4wsstc5chcecanngsxnq65ibcxfj5fokkuszhkhkforf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bhvzlpbihqdo36tu5pb3rvhvq62o2xwey5zkqm22tg4e7cexht.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 320
    x2 = (xindex // 20480)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/ty/cty7kjahvyjnsyiw6kcs7w6h6dcytuzxezbrfzkpqdmnjwgr7gu2.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (12288*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (118784 + x2 + (64*y0) + (131072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (118784 + x2 + (64*y0) + (131072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (118784 + x2 + (64*y0) + (131072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (118784 + x2 + (64*y0) + (131072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4uokyf5mqdvl5xlh7mpyldk4bsjmu4patxwhhbc36b4waf3x4s.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (12288*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccb7qysy5x3a2ucdtqphxjpvcnifg3lc6padhaxukrdovybqemtt.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (12288*(r2 // 64)) + (24576*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czuhj6fwp3jcxhorpdcjelita5buq6hxhtvfmdc2lu4bixjz6cyr.py
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
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (12288*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvjxncciooct67pap3rekg3w7jkm5r7ljmqxjzv2ytkh5cetjro.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (24576*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (94208 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (94208 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (94208 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (94208 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cps53d5yrknpp5r2etmg7a2r735kpofr5a6mwqskxsloo75bpr5o.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (24576*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvjovqyt26ereqywayr357e6aidedph3zg67dhagpau7cqsksfe.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (24576*(r2 // 64)) + (49152*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3itgwlzhmvnvnaykvj2jdsnpnucw4xh36qc4756pfcsevysq6s.py
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
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (24576*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirnfobwhgzdw34jcifliac2dpo7rplst3ihfh53mhquyrsm3zlf.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (24576*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (69632 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (69632 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (69632 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (69632 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7f2rsfzkx7fxnct2jzcn2hzwxpgktkjgacdznfwywubwvisuvj.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (24576*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (45056 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (45056 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (45056 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (45056 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwd25kksgrcbd6zfwadltm6g3mk4iap3wz6tq463uxxny6vckhox.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (24576*y1)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (20480 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (20480 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (20480 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (20480 + x2 + (64*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfpwxr4nobouctw64wuwat357p6uabjbh3uaz3q7wzrshji5dihh.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (320*x2) + (20480*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y0) + (131072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (64*y0) + (131072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (64*y0) + (131072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (64*y0) + (131072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yl/cyldcv62zj5bmdytajpzcn4g22kt4p6ldct5um6n3mfamovfh4ek.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 320
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crzkf6xfyseoymvrc243iez72ndy4ak5itnu4cexx46rhize6aqj.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (20480*(r2 // 64)) + (40960*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgvpwyqen2g6wtjdvosgsy3kh5wauwl4kzni3jrbciie63nfow6.py
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
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 320
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (20480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/csh5yscq5mfhkjnyb6x5mtyfe3pdrz75cowia65izslkizayyjfs.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward]

triton_poi_fused_add_avg_pool2d_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 8
    x2 = (xindex // 64)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + ((8*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None)
    tmp25 = tl.load(in_ptr0 + ((8*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + ((8*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + ((8*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None)
    tmp42 = tl.load(in_ptr0 + ((8*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + ((8*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + ((8*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(8, 2 + x1))))) + (64*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(8, 2 + x0))))), None)
    tmp59 = tl.load(in_out_ptr0 + (x4), None)
    tmp61 = tl.load(in_ptr1 + (x4), None)
    tmp63 = tl.load(in_ptr2 + (x4), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(8, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(8, 2 + x0)
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
    tmp60 = tmp58 + tmp59
    tmp62 = tmp60 + tmp61
    tmp64 = tmp62 + tmp63
    tl.store(in_out_ptr0 + (x4), tmp64, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbm7rax2czm4wiw57s22pgnpvk7npnxpb7eyaagqb3z52gtqlsw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (20480 + (64*x0) + (81920*(r2 // 64)) + (163840*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7a/c7acqcjqgid7mknujrecogykgrovf5fnwqclsnswia4vjbwg3kn6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (20480 + y0 + (64*x2) + (81920*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.001953125
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp545oy7eudajbzhh7yqwtozt45eie7zcofyqsgumsk2effkdl37.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((289*x1) + (55488*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3l/c3lj7ovt3i5krwspxrmbi6fwojv44cxbzxnwigoiebgejkxxdjpi.py
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
    size_hints=[256, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crphnpvse4pfh5tav4x7oonagrfp7sjmyzpqigwvbtaqmahb4pf6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((289*x0) + (55488*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvhhuvmnbdosfsqargcl624bganjz25ylav23hahee73o3k4mlh.py
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
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmhzsnfeekpiklq27a4myuznkkae5dxu2uiy5uy7gdrset6yte7.py
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
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (289*x2) + (55488*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00043252595155709344
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ab/caby7xi7p7wb7sh6gsweybeatckneabjfds2h2twy6sngkioomti.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((64*x0) + (81920*(r2 // 64)) + (163840*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (320*r2) + (40960*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ya/cyapzjrsu2np5mbeqyflumsnjakvwv4w5r5sjaxk4bx5um5a3elf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 320
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (64*x2) + (81920*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.001953125
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtbb6buj7kuy2rroobaqmq2nhkprqofkq2sluvtnxcqf24r5x5v.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (576 + x1 + (768*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (166464 + (289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (166464 + (289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ra/crapbxxhm7w4mi7jcs7stgpm7p4yk3df4q5xdps73mg5e543cyvc.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (576 + x0 + (768*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (166464 + (289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (166464 + (289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.load(in_ptr4 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqmvadjmcdhbh7rvckm64ncfvxy4quu2ngqv5mhfsrb7qqbbouo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (576 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (166464 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (166464 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00043252595155709344
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6vnl33h7phyhunmunjqbf37onfw27b5gtyr2pc3ncwcxr3jxki.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1775616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 17
    x1 = (xindex // 17) % 17
    x2 = (xindex // 289)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((17*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((17*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + ((17*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None)
    tmp25 = tl.load(in_ptr0 + ((17*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + ((17*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + ((17*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None)
    tmp42 = tl.load(in_ptr0 + ((17*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + ((17*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + ((17*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(17, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(17, 2 + x0)
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
    tl.store(out_ptr0 + (x4), tmp58, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2ttg6ifvxykew2n3zvcwymi33tgmkuqnmh3qzy7bobwh3uieqe.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (384 + x1 + (768*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (110976 + (289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (110976 + (289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c643peihjnxa42fvyp6kraiuuyrgjzqv4eerkopbfkgdsgbjckxr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (384 + x0 + (768*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (110976 + (289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (110976 + (289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.load(in_ptr4 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnoqxc4tk4yrmtprj2wnmys57basjv4kkteb6vuenrkl633yguc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (384 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (110976 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (110976 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00043252595155709344
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4pe6ptlmjahmpq6umltqtw6uhkksevgzyvemhsten5g76koo25.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (192 + x1 + (768*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (55488 + (289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (55488 + (289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7i/c7iie2rzyatv5o7h4447rxyunft5m3egzoxmfi3pg2ifnjpkdbgq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (192 + x0 + (768*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (55488 + (289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (55488 + (289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.load(in_ptr4 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/celkzxktzhzkai7virvyr2idszyraaejoj3a4ndf4joiuehusj65.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (192 + x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (55488 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (55488 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00043252595155709344
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6hwtxzsca52lfsnag2pbeatobygty34mb2smmb34sewkfh2ubg2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (x1 + (768*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + ((289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6p5hqns2kii7jyobtvbtmrae5vuha4f2ntbrmfyt6nsy7hnpsad.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + ((289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.load(in_ptr4 + (x0 + (192*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ie/ciefcz6rvult4n2yzdpezio5qinv7pmyjdeauerfb4cypudtbyj7.py
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
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00043252595155709344
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wz/cwztsne5z5ueu562nijkwdpsmzdxqnbrzadyfzyrtp5y3r4ozzcv.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (55488*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (166464 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (166464 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (166464 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (166464 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (289*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4iqkvz62hyijtjt6naynjpiw5uel7hwv74jqqsl6qhdllnm6fr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 2312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 289
        r2 = (rindex // 289)
        tmp0 = tl.load(in_ptr0 + (r1 + (289*x0) + (55488*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chlgvkcxtt3tf5c76ierxtvypvqql7r5zrwqd7myxj7t24uugphs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3648
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((289*x1) + (55488*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (192*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkb752qkosd2jfbj3gspxlehncnvyupizqxjiguxnte2llg7on4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ez/cezs7w5dejra4h23ih2ptp4jia3mk6msqsveh5jr6eupax5pgk2o.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (289*x2) + (55488*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00043252595155709344
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ka/ckahtxuwgeags6ncuvb4ya2w7vfnlypmwoqqrybmtheicfwzdgza.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (55488*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (110976 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (110976 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (110976 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (110976 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (289*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xs/cxsshwwouhtl2mlqjmn3hsduk2dscy6k3ma3efdf5e5uqwnqcw3e.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3040
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (160*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((289*x1) + (46240*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vtjpsvfy3hrjekxcah266bjeqtmwdglgh5dwdingwdxn3377fy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfatpd3lrncxz6murq4limlbc5jp2uo446tieoeyu6vpeadoc4td.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3040
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 160)
    x0 = xindex % 160
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (160*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((289*x0) + (46240*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (160*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb5takeylczfo52v4tvh7n4ccjchjj5frd5zhtk3ysmvhhf3s57k.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clrw4r77xx7rwaseye2c4tgihcm4yf5ocamby2ysfttnsaolaaxq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 160
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (289*x2) + (46240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00043252595155709344
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (160*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dcbufqybyznkmz4zkvjayhsioeblrbbe3eso7rbalfscq42zkg.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (55488*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (55488 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (55488 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (55488 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (55488 + x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (289*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xuutrik6degsgg6yuyodke7am5jo6o3hts4wqxjmcenrorhnbt.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (55488*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (289*y0) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (289*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c27jbsnefnruri7et5d27cngnngjcsfxdcrd5ddghacoom4ry5ad.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2432
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (128*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((289*x1) + (36992*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jd/cjdvamzq2qxypfddttale6frmaqlo5uysxsni2silh4se7eusvjf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5wt5eo5seyrgvzfo47xhykzfkgliokbg7qavrcwr33nifuz3xr.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2432
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((289*x0) + (36992*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (128*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77wzwchorgwk46x4dylduagyl5vzb7ldccul6zlmyaxbkeoitgd.py
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
    size_hints=[128, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 19
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/ox/coxkmvzeui4syl42uqipw4hoe2n6ewodv32a4rdbjjd5jdv4jmgg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (289*x2) + (36992*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00043252595155709344
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cnejgxwhwhwzn5asjn222bipy2oumvf4pamhmhf3coyo25rxb7fr.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward]

triton_poi_fused_add_avg_pool2d_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1775616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 17
    x1 = (xindex // 17) % 17
    x2 = (xindex // 289)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((17*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((17*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + ((17*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None)
    tmp25 = tl.load(in_ptr0 + ((17*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + ((17*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + ((17*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None)
    tmp42 = tl.load(in_ptr0 + ((17*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + ((17*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + ((17*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(17, 2 + x1))))) + (289*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(17, 2 + x0))))), None)
    tmp59 = tl.load(in_ptr1 + (x4), None)
    tmp61 = tl.load(in_ptr2 + (x4), None)
    tmp63 = tl.load(in_ptr3 + (x4), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(17, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(17, 2 + x0)
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
    tmp60 = tmp58 + tmp59
    tmp62 = tmp60 + tmp61
    tmp64 = tmp62 + tmp63
    tl.store(in_out_ptr0 + (x4), tmp64, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dz/cdzaqgpl6sy4krw6dfsczbvgjrex3yzq2bxj23yot5osgogir4zg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1824
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (96*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (110976 + (289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/cawj3anhk5a2pmxr2hgmne3ieqwrmgkrc5jm6kbdomubs5izomfc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bydstcd2pp3juhbkyrqbdhfa5vzijdgaprvcf5vixrtkpewjy5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1824
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (110976 + (289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (96*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zteqqkn6r4zezgjcbnnwq5qkpxcq5opp2v6lmnjfnd5qmr5whj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cje7fp4wdgwbp47ag5fpd5o7iu7l62t54ozimyzhit5yeokgsvoy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (110976 + y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00043252595155709344
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cje35oblfphwltf7vndbdrvbmoifjj24wybflukrsflettrqr3pp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (96*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((1225*x1) + (117600*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcfqws7pl4vqbjbhazwi2mpxgpa4huqyoickmtxqg4a4zqnfog3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 77
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (77*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czvy6ursk76antbleiowclkodqpili54eu7zvpxfjywqdqjsjdj2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((1225*x0) + (117600*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (96*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mw/cmwdppfzrsgwj54y4hl3rownarq4arvob6n76zey3avw6yxgfqjs.py
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
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 77
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggahan6psuqq3m466uvbr67hpdehha6wg35rvjwbv3zgyfwwunj.py
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
    size_hints=[16384, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1225*x2) + (117600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00010204081632653062
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oo/coomqcg6lnwmhp6w47bv453vfz5ih7flm2opvvqcysw77qaxubmv.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((1225*x1) + (78400*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mu/cmugm6ex2lxgfckgq3tu3gri5q3jrr4rmpmhz4tln4cxi3lkpeym.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 77
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (77*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/coraqvhioizfkxllx7qvbxok5ld2gpycqe2x6pjkyemocqimzvm7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((1225*x0) + (78400*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gu/cgua7piphp5s6qac3amwibgtxvl6zb4gepkuxcjr2qzxcbjnarv4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 77
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7zugq4ko3lisxbbmlx5t4prg3x346wjc22tauksfmov73fzjzh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1225*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00010204081632653062
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lg/clglorlwfg2rmestzo46553jtifzprhrzlp223wkvcbxjx726wvl.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7296
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 19
    x1 = (xindex // 19)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x0)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (384*((r2 + (122*x0)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((289*x1) + (221952*(((r2 + (122*x0)) // 289) % 8)) + ((r2 + (122*x0)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tq66p3yr6obuay6v25kbkv6zwy7dpkl5aq4cmatwqmtaov4lhm.py
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
    size_hints=[512, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (19*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a6/ca6qrmvmoup2ydp2ij2gfscsyfstu3barq2chkx3fmcvqclr5ggi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7296
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 2312, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((289*x0) + (221952*(((r2 + (122*x1)) // 289) % 8)) + ((r2 + (122*x1)) % 289)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (384*((r2 + (122*x1)) % 2312))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq42znnq43qdb73avznkopeaxhm6ze7oqbcnnodfhxmanc7ejdgm.py
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
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 19
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5iwumxgxiad2jgcfezqzktyvubvqa4gjqxiwn4ih22qzx3nhcx.py
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
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2312
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 289
    y1 = (yindex // 289)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (289*x2) + (221952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00043252595155709344
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yu/cyufvrh6m7a5f2px3e7g4yxush4rpkeuzyenp7tiu37yeifq2rnm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (224 + x1 + (288*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (274400 + (1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (274400 + (1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmyhy5ja5nemacethjkqynt34zyveej3ov5m32yrbo2qg6dxfzwb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (224 + x0 + (288*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (274400 + (1225*x0) + (352800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (274400 + (1225*x0) + (352800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.load(in_ptr4 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2s/c2swalqmn627bf4pvsxpz3k2o6jbewj4dpoo37yqtivlqzljxxkp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_106', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (224 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (274400 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (274400 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00010204081632653062
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ra/crauww57h7vdxx52tgb5yxiu4dslicbtcvxnyikynhetrkgsg4lk.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2822400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 35
    x1 = (xindex // 35) % 35
    x2 = (xindex // 1225)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp25 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp42 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(35, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(35, 2 + x0)
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
    tl.store(out_ptr0 + (x4), tmp58, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73bvcitshraaupob3au7vd4alumdnshvbl6cdjocpe2jh5i6wdk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (96*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (128 + x1 + (288*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (156800 + (1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (156800 + (1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch2jtcodcdbtqptn5urtam3wsnfbl2vzfoamktvyyylhdlyvgvhf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_109', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (128 + x0 + (288*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (156800 + (1225*x0) + (352800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (156800 + (1225*x0) + (352800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.load(in_ptr4 + (x0 + (96*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u6/cu6zl4ziuk2puqdri6j3imubyrf5y562rgo24zjktpwh6q53fe3p.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_110', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (128 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (156800 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (156800 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00010204081632653062
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (96*y3)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqieludh36psnetl56v6xhutzgggigyk2mhv3cxm2czvwi3dkyzn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_111 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (64 + x1 + (288*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (78400 + (1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (78400 + (1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/cswesvjddrmg3tz7d3fc2e35vlhqxm3kapy3nwq7eds4gwmjsrp4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_112 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_112', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (64 + x0 + (288*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (78400 + (1225*x0) + (352800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (78400 + (1225*x0) + (352800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.load(in_ptr4 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck66poyymo2enmwsjwumgkkou6q6gcgmmi2rupcfy5zxyxwb4pnw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_113 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_113', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (64 + x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (78400 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (78400 + y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00010204081632653062
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbn26rfuc2it7cqtbjqgtjxkq3as2qsx7sdk7b5ashden5kztsob.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3696
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (48*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((1225*x1) + (58800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7x2oglfkioowl4in66ulnjkssei3cxghbdrwepfakstnj76hsrb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 77
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (77*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c566yileq7z7bvx5m7f4azjjf3tsrsb3jp5wwn75xpdeejmaokgv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_116 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3696
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 48)
    x0 = xindex % 48
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (48*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((1225*x0) + (58800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (48*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7uqppps4smbx2an3jmapihgztvung67tetjmbgcta4cv5dkbmk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_117 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 77
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfb4uadbzvjor6gavaaijhr5djqlsahpk7ergrgaaxre4qhtj6mi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_118 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_118', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 48
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (48*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1225*x2) + (58800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (48*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00010204081632653062
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (48*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l3/cl35jicgeqqwuszrxxjf7yimic6fnyj6lfh7ndpid4rtkoj534b5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_119 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (x1 + (288*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + ((1225*x1) + (352800*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/ciry5e2hoegiabes423yiziczxvrvqzohav6w3veuzbwdnvwbysn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_120 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (x0 + (288*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((1225*x0) + (352800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + ((1225*x0) + (352800*(((r2 + (128*x1)) // 1225) % 8)) + ((r2 + (128*x1)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tl.load(in_ptr4 + (x0 + (64*((r2 + (128*x1)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvsietf46qvxz2t4gwpftwqvf2qzap32vwakkew655l7nw7ibu6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_121 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_121', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (288*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0 + (1225*x2) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp0, tmp6, tmp5)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.00010204081632653062
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqas6u7fwfadvzg2kz5w337foafa2l4fodc6o4ci4b4nt5obpvfs.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_122 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_122', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (274400 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (274400 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (274400 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (274400 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (1225*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaawenfdqsimduffu6c4znm54cxsfcldig4br724fubh3322w5m.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((1225*x0) + (78400*(r2 // 1225)) + (313600*x1) + (r2 % 1225)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cecvm3c7lmi27vw4lbmevpwedxzx5lsdzbp42olqcbvkecjqfrio.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_124 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_124', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/rk/crkt7llbknaerj62tv6bokrocz3pacgn3zgnurv7mgpxhnhxbzzo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_125', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4928
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((1225*x1) + (78400*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (64*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vt2kvaojm7usc475uhy7ymwqdvawgppqpkjh62ern7itfhvfhs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_126 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 77
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (77*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktv2gq3m64wwfdzqo34ndytbyx5eb7oxjj6hhocz32tf2pqy4sy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_127', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1225*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00010204081632653062
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3tbfgkqxyowccho3res75ez52slk47d7j4qqukiwnu4egb5ah2.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_128', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 35
    x1 = (xindex // 35) % 35
    x2 = (xindex // 1225)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), None)
    tmp25 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), None)
    tmp42 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(35, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(35, 2 + x0)
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
    tl.store(out_ptr0 + (x4), tmp58, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crboix63qri3hzvd6yw5cxrhc3464rlynrpiixjeq6765myijurk.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_129 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (117600*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (156800 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (156800 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (156800 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (156800 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (1225*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33rkkaetcjpq3j2d6qexgglf63aemrcieijr65nn3wmfpj363zu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_130 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 4900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((1225*x0) + (117600*(r2 // 1225)) + (470400*x1) + (r2 % 1225)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvw7cnkl35hjkmhhcnt74zf4xhkr2d2w7vjk4khhqbgs33skodg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_131 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_131', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3p7ky26rjivbbriiy27zrgz6kxkewhem5kdqar2ux66qoxqq23.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_132 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((1225*x1) + (117600*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (96*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czah7kp7z3frk7gsum2stfh3ciynxla7n35wolcba2tn744gvdrl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_133 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_133', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 77
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (77*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7n/c7nojprityzkwhhkhuz64egwqi3a2p7x2tuinzp44nebr5gfi4xp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_134 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_134', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1225*x2) + (117600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00010204081632653062
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hj/chjnfdlzbohgxgjtgqawjtp675aclhy752dzdfvj2336iuwjal4w.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_135 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_135', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (78400 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (78400 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (78400 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (78400 + x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (1225*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3idva52unjay2azq5xufpbdkszaimis4te6fcs7ebu5v6ccymj.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_136 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_136', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (1225*y0) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (1225*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csb5ot2qbvtsk3ycxbebuaty42x5ukp6wycz4ou6jtpxrcjkxtr3.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_137 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_137', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x2) + (39200*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (274400 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (274400 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (274400 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (274400 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (1225*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qt/cqt5xfbizvxrtu6npuk3xjhbyo5gmmt4pzqvja2kkyzq25mpdtyp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_138 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_138', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((1225*x0) + (39200*(r2 // 1225)) + (156800*x1) + (r2 % 1225)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co6n34n7kq4zehcnsvlifnrhhyi2ob7oh5kgrdy6dpxvrepz2lhu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_139 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_139', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/qi/cqi5fwy7d2eclrmepokpibydsxz3mh6oxrqw5exzggsxaav426z2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_140', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2464
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 77
    x1 = (xindex // 77)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 9800, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((1225*x1) + (39200*(((r2 + (128*x0)) // 1225) % 8)) + ((r2 + (128*x0)) % 1225)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (32*((r2 + (128*x0)) % 9800))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4wgqrt3xkxmx6rg2iny4tbhxv55gamw6eny6h5bf3yuzhhbirxz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_141 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_141', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 77
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (77*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3v7alflco2x2xbaldpgwdwu3b6ocm5faszk7iih23t3euvjwogy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_142 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_142', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9800
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1225
    y1 = (yindex // 1225)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1225*x2) + (39200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00010204081632653062
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/cconrcvg3o3rzpmryw2n3hxpe3tkex2lhzhwogab32twosraudqw.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_143 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_143', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (117600*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (156800 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (156800 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (156800 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (156800 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (1225*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tp/ctp2jx23rlee5qgccw6untubtsubkqjxkufyrfvowzjblct2digz.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_144 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_144', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (78400 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (78400 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (78400 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (78400 + x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (1225*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvglfrhwb4sutpx4w5ghykucojyvirat3xdu2ivd2fhprnzcgswx.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_145 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_145', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (78400*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (1225*y0) + (313600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp0, tmp8, tmp7)
    tl.store(out_ptr0 + (x2 + (1225*y3)), tmp9, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6ui5sfss42cijxgiyu64mqztnpillzuqm2dsv3pyar6u6sscjm.py
# Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward]

triton_poi_fused_add_avg_pool2d_backward_146 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_146', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 35
    x1 = (xindex // 35) % 35
    x2 = (xindex // 1225)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + ((35*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp25 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + ((35*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp42 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + ((35*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(35, 2 + x1))))) + (1225*x2) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(35, 2 + x0))))), xmask)
    tmp59 = tl.load(in_ptr1 + (x4), xmask)
    tmp61 = tl.load(in_ptr2 + (x4), xmask)
    tmp63 = tl.load(in_ptr3 + (x4), xmask)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(35, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(35, 2 + x0)
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
    tmp60 = tmp58 + tmp59
    tmp62 = tmp60 + tmp61
    tmp64 = tmp62 + tmp63
    tl.store(in_out_ptr0 + (x4), tmp64, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yv/cyvy6z2idw7z4zuxgq6n6vjznirqkay3v4neidmlzo52hj2wabcp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_147 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_147', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 60672
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 40328, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (128*x1)) % 40328))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (192*((r2 + (128*x1)) % 40328))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (192*((r2 + (128*x1)) % 40328))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tu/ctuvfjw6hmxothnwanv3zpjc7uu4mkalmypruybhigz3aoi5q7xh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_148 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_148', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 316
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cne2kv6r5ysvro3xx4ehbgu4n6y4ckvdgs5mvgrzdjfjd3txhs2a.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_149 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_149', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 316
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/czijpinezv7ovshmlfrhgvdcoeuervutmaa3flycljnkmuknufcz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_150 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_150', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7742976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 2.479666732791113e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcvvlma5fm6dc5zryi3ooeubxdayhrfm67tlgz7y5ggtxedun5g.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_151 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_151', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 26720
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 334
    x1 = (xindex // 334)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 42632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (80*((r2 + (128*x0)) % 42632))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((5329*x1) + (426320*(((r2 + (128*x0)) // 5329) % 8)) + ((r2 + (128*x0)) % 5329)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bc/cbcyd2exynjiw4v7smlueb6a2xwcaslcmhridj2nsjtssqhnoucd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_152 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_152', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 80
    XBLOCK: tl.constexpr = 1
    rnumel = 334
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (334*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dc/cdcaghofc5ppjej6yka3dikt2y2s5wnlj3hj46s6vseslsnnrhzn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_153 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_153', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 26720
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 80)
    x0 = xindex % 80
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 42632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (80*((r2 + (128*x1)) % 42632))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((5329*x0) + (426320*(((r2 + (128*x1)) // 5329) % 8)) + ((r2 + (128*x1)) % 5329)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (80*((r2 + (128*x1)) % 42632))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crz25t5pkfsox7kqslhuxtzlop2iswzkecrb66v72kxf36ju6zaa.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_154 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_154', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 334
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (80*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4zuqciimtqspxwsyh6m6gbucccq3dlxesfykqbqeecryebpzxb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_155 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_155', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 42632
    xnumel = 80
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 5329
    y1 = (yindex // 5329)
    tmp0 = tl.load(in_ptr0 + (x2 + (80*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (5329*x2) + (426320*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (80*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 2.3456558453743668e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (80*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3hamqzimmjt6ka5ej7nadq2bswkpldefie6xlpguy3unmrgk6r.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_156 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_156', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65856
    rnumel = 168
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (10752*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (64*r2) + (10752*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (64*r2) + (10752*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jo/cjo2o3ayljt4gjvxowetnbapaciejdt6oz5vm4vxyhfnqp42zojt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_157 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_157', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 1029
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/csev5prt6vvjrrep5e27bwp36dsvj2rv6nsw554kowta3cmg7rmq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_158 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_158', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 1029
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdeexnijjmavxs4fcggjojvtbvvd73zykguvutamxxp4ig3sjcie.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_159 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_159', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11063808
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
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 5.78462677588042e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l3/cl3jjokfrqvfmqqwtnieqvcjxfv4xt356d3v4gpi2fh5yl4d7bii.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_160 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_160', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43232
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1351
    x1 = (xindex // 1351)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 172872, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (32*((r2 + (128*x0)) % 172872))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((21609*x1) + (691488*(((r2 + (128*x0)) // 21609) % 8)) + ((r2 + (128*x0)) % 21609)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjv63g3pnoxwgpvnhlkuulor35lidii37l6u7ib5463goszmjrt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_161 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_161', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 1351
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1351*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnv2o7xgrvu2xmfguxvmf3s7d3c27tla64bijrrdtliqpq2mpwi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_162 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_162', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43232
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 172872, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (128*x1)) % 172872))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((21609*x0) + (691488*(((r2 + (128*x1)) // 21609) % 8)) + ((r2 + (128*x1)) % 21609)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (32*((r2 + (128*x1)) % 172872))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/ceceynugzm2zzlxeg3vhpvwhdrpasmsglwlvdwhds2tq2tgnkpf5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_163 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_163', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 1351
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ak/cakn4bnyd4s2l7tsecbdx6k7dk42qyxkqv3sgyedkp2b4ycwyu4f.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_164 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_164', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 172872
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 21609
    y1 = (yindex // 21609)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (21609*x2) + (691488*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 5.78462677588042e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcgne2n6vm7ycn6me7s55ee7r7bnu6ertnieocizfbod7m6tw3v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_165 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_165', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 44416
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1388
    x1 = (xindex // 1388)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x0)
        tmp1 = tl.full([1, 1], 177608, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (32*((r2 + (128*x0)) % 177608))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((22201*x1) + (710432*(((r2 + (128*x0)) // 22201) % 8)) + ((r2 + (128*x0)) % 22201)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cux7opgrgetsq2ednv57p7rajzoyhwr2wymltpjmshx5krbzvt5t.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_166 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_166', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 1388
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1388*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cenz3yxerjye7pew4g4rxh37mp5yvffxvr2apjnwktj7kzhepd3s.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_167 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_167', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 44416
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 177608, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (128*x1)) % 177608))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((22201*x0) + (710432*(((r2 + (128*x1)) // 22201) % 8)) + ((r2 + (128*x1)) % 22201)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (32*((r2 + (128*x1)) % 177608))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4y/c4yunvr5b2bxqsfv2e4bstzwjeygfibx2jxxbgxblui3vugkib3x.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_168 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_168', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 1388
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vd/cvd4gycrjm3b4jx2fby5gvelez5jbvcpksnlczhn37yq34bu4tbn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_169 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_169', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 177608
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 22201
    y1 = (yindex // 22201)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (22201*x2) + (710432*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 5.630377010044593e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp21, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_567, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_1, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_2, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_3, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_4, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_5, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_6, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, relu_73, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, getitem_159, cat_8, convolution_76, squeeze_229, convolution_77, squeeze_232, relu_77, convolution_78, squeeze_235, convolution_79, squeeze_238, convolution_80, squeeze_241, relu_80, convolution_81, squeeze_244, relu_81, convolution_82, squeeze_247, convolution_83, squeeze_250, avg_pool2d_7, convolution_84, squeeze_253, cat_11, convolution_85, squeeze_256, convolution_86, squeeze_259, relu_86, convolution_87, squeeze_262, convolution_88, squeeze_265, convolution_89, squeeze_268, relu_89, convolution_90, squeeze_271, relu_90, convolution_91, squeeze_274, convolution_92, squeeze_277, avg_pool2d_8, convolution_93, squeeze_280, clone, permute_1, le, unsqueeze_378, le_1, unsqueeze_390, le_2, unsqueeze_402, unsqueeze_414, unsqueeze_426, le_5, unsqueeze_438, le_6, unsqueeze_450, unsqueeze_462, le_8, unsqueeze_474, le_9, unsqueeze_486, le_10, unsqueeze_498, le_11, unsqueeze_510, unsqueeze_522, unsqueeze_534, le_14, unsqueeze_546, le_15, unsqueeze_558, unsqueeze_570, le_17, unsqueeze_582, le_18, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, le_22, unsqueeze_642, unsqueeze_654, le_24, unsqueeze_666, le_25, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, le_30, unsqueeze_738, unsqueeze_750, unsqueeze_762, le_33, unsqueeze_774, le_34, unsqueeze_786, le_35, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_40, unsqueeze_858, unsqueeze_870, unsqueeze_882, le_43, unsqueeze_894, le_44, unsqueeze_906, le_45, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, le_50, unsqueeze_978, unsqueeze_990, unsqueeze_1002, le_53, unsqueeze_1014, le_54, unsqueeze_1026, le_55, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, unsqueeze_1074, unsqueeze_1086, le_60, unsqueeze_1098, unsqueeze_1110, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, unsqueeze_1230, le_72, unsqueeze_1242, unsqueeze_1254, le_74, unsqueeze_1266, le_75, unsqueeze_1278, le_76, unsqueeze_1290, unsqueeze_1302, unsqueeze_1314, le_79, unsqueeze_1326, unsqueeze_1338, le_81, unsqueeze_1350, le_82, unsqueeze_1362, le_83, unsqueeze_1374, unsqueeze_1386, unsqueeze_1398, le_86, unsqueeze_1410, unsqueeze_1422, le_88, unsqueeze_1434, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1482, unsqueeze_1494, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (80, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (48, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_41, (48, ), (1, ))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_47, (96, ), (1, ))
    assert_size_stride(primals_49, (96, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_53, (384, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_57, (96, ), (1, ))
    assert_size_stride(primals_59, (96, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_67, (192, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_77, (192, ), (1, ))
    assert_size_stride(primals_79, (192, ), (1, ))
    assert_size_stride(primals_81, (192, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_85, (160, ), (1, ))
    assert_size_stride(primals_87, (192, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_91, (160, ), (1, ))
    assert_size_stride(primals_93, (160, ), (1, ))
    assert_size_stride(primals_95, (160, ), (1, ))
    assert_size_stride(primals_97, (192, ), (1, ))
    assert_size_stride(primals_99, (192, ), (1, ))
    assert_size_stride(primals_101, (192, ), (1, ))
    assert_size_stride(primals_103, (160, ), (1, ))
    assert_size_stride(primals_105, (160, ), (1, ))
    assert_size_stride(primals_107, (192, ), (1, ))
    assert_size_stride(primals_109, (160, ), (1, ))
    assert_size_stride(primals_111, (160, ), (1, ))
    assert_size_stride(primals_113, (160, ), (1, ))
    assert_size_stride(primals_115, (160, ), (1, ))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_119, (192, ), (1, ))
    assert_size_stride(primals_121, (192, ), (1, ))
    assert_size_stride(primals_123, (192, ), (1, ))
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_127, (192, ), (1, ))
    assert_size_stride(primals_129, (192, ), (1, ))
    assert_size_stride(primals_131, (192, ), (1, ))
    assert_size_stride(primals_133, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_137, (192, ), (1, ))
    assert_size_stride(primals_139, (192, ), (1, ))
    assert_size_stride(primals_141, (192, ), (1, ))
    assert_size_stride(primals_143, (320, ), (1, ))
    assert_size_stride(primals_145, (192, ), (1, ))
    assert_size_stride(primals_147, (192, ), (1, ))
    assert_size_stride(primals_149, (192, ), (1, ))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_153, (320, ), (1, ))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_161, (448, ), (1, ))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_169, (192, ), (1, ))
    assert_size_stride(primals_171, (320, ), (1, ))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_179, (448, ), (1, ))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_185, (384, ), (1, ))
    assert_size_stride(primals_187, (192, ), (1, ))
    assert_size_stride(primals_189, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_190, (32, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_191, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_192, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_193, (192, 80, 3, 3), (720, 1, 240, 80))
    assert_size_stride(primals_194, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_195, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_196, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_197, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_198, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_199, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_200, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_201, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_202, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_203, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_204, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_205, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_206, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_207, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_208, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_209, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_210, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_211, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_212, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_213, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_214, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_215, (384, 288, 3, 3), (2592, 1, 864, 288))
    assert_size_stride(primals_216, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_217, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_218, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_219, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_220, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_221, (128, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_222, (192, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_223, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_224, (128, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_225, (128, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_226, (128, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_227, (192, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_228, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_229, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_230, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_231, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_232, (192, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_233, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_234, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_235, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_236, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_237, (192, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_238, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_239, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_240, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_241, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_242, (192, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_243, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_244, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_245, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_246, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_247, (192, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_248, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_249, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_250, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_251, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_252, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_253, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_254, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_255, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_256, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_257, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_258, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_259, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_260, (320, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_261, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_262, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_263, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_264, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_265, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_266, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_267, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_268, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_269, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_270, (384, 448, 3, 3), (4032, 1, 1344, 448))
    assert_size_stride(primals_271, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_272, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_273, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_274, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_275, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_276, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_277, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_278, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_279, (384, 448, 3, 3), (4032, 1, 1344, 448))
    assert_size_stride(primals_280, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_281, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_282, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_567, (8, 3, 299, 299), (268203, 1, 897, 3))
    assert_size_stride(convolution, (8, 32, 149, 149), (710432, 1, 4768, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 149, 149), (710432, 1, 4768, 32))
    assert_size_stride(convolution_1, (8, 32, 147, 147), (691488, 1, 4704, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(relu_1, (8, 32, 147, 147), (691488, 1, 4704, 32))
    assert_size_stride(convolution_2, (8, 64, 147, 147), (1382976, 1, 9408, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 147, 147), (1382976, 1, 9408, 64))
    assert_size_stride(getitem_6, (8, 64, 73, 73), (341056, 1, 4672, 64))
    assert_size_stride(getitem_7, (8, 64, 73, 73), (341056, 1, 4672, 64))
    assert_size_stride(convolution_3, (8, 80, 73, 73), (426320, 1, 5840, 80))
    assert_size_stride(squeeze_10, (80, ), (1, ))
    assert_size_stride(relu_3, (8, 80, 73, 73), (426320, 1, 5840, 80))
    assert_size_stride(convolution_4, (8, 192, 71, 71), (967872, 1, 13632, 192))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(relu_4, (8, 192, 71, 71), (967872, 1, 13632, 192))
    assert_size_stride(getitem_12, (8, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(getitem_13, (8, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(convolution_5, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(convolution_6, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_19, (48, ), (1, ))
    assert_size_stride(relu_6, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_7, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(convolution_8, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_8, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_9, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_28, (96, ), (1, ))
    assert_size_stride(relu_9, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_10, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_31, (96, ), (1, ))
    assert_size_stride(avg_pool2d, (8, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(convolution_11, (8, 32, 35, 35), (39200, 1, 1120, 32))
    assert_size_stride(squeeze_34, (32, ), (1, ))
    assert_size_stride(cat, (8, 256, 35, 35), (313600, 1, 8960, 256))
    assert_size_stride(convolution_12, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_37, (64, ), (1, ))
    assert_size_stride(convolution_13, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_40, (48, ), (1, ))
    assert_size_stride(relu_13, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_14, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_43, (64, ), (1, ))
    assert_size_stride(convolution_15, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_46, (64, ), (1, ))
    assert_size_stride(relu_15, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_16, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_49, (96, ), (1, ))
    assert_size_stride(relu_16, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_17, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_52, (96, ), (1, ))
    assert_size_stride(avg_pool2d_1, (8, 256, 35, 35), (313600, 1, 8960, 256))
    assert_size_stride(convolution_18, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_55, (64, ), (1, ))
    assert_size_stride(cat_1, (8, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_19, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_58, (64, ), (1, ))
    assert_size_stride(convolution_20, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_61, (48, ), (1, ))
    assert_size_stride(relu_20, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_21, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_64, (64, ), (1, ))
    assert_size_stride(convolution_22, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_67, (64, ), (1, ))
    assert_size_stride(relu_22, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_23, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_70, (96, ), (1, ))
    assert_size_stride(relu_23, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_24, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_73, (96, ), (1, ))
    assert_size_stride(avg_pool2d_2, (8, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_25, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_76, (64, ), (1, ))
    assert_size_stride(cat_2, (8, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_26, (8, 384, 17, 17), (110976, 1, 6528, 384))
    assert_size_stride(squeeze_79, (384, ), (1, ))
    assert_size_stride(convolution_27, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_82, (64, ), (1, ))
    assert_size_stride(relu_27, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_28, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_85, (96, ), (1, ))
    assert_size_stride(relu_28, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_29, (8, 96, 17, 17), (27744, 1, 1632, 96))
    assert_size_stride(squeeze_88, (96, ), (1, ))
    assert_size_stride(getitem_65, (8, 288, 17, 17), (83232, 1, 4896, 288))
    assert_size_stride(cat_3, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_30, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_91, (192, ), (1, ))
    assert_size_stride(convolution_31, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_94, (128, ), (1, ))
    assert_size_stride(relu_31, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_32, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_97, (128, ), (1, ))
    assert_size_stride(relu_32, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_33, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_100, (192, ), (1, ))
    assert_size_stride(convolution_34, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_103, (128, ), (1, ))
    assert_size_stride(relu_34, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_35, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_106, (128, ), (1, ))
    assert_size_stride(relu_35, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_36, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_109, (128, ), (1, ))
    assert_size_stride(relu_36, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_37, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_112, (128, ), (1, ))
    assert_size_stride(relu_37, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_38, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_115, (192, ), (1, ))
    assert_size_stride(avg_pool2d_3, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_39, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_118, (192, ), (1, ))
    assert_size_stride(cat_4, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_40, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_121, (192, ), (1, ))
    assert_size_stride(convolution_41, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(relu_41, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_42, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_127, (160, ), (1, ))
    assert_size_stride(relu_42, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_43, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_130, (192, ), (1, ))
    assert_size_stride(convolution_44, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(relu_44, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_45, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_136, (160, ), (1, ))
    assert_size_stride(relu_45, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_46, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_139, (160, ), (1, ))
    assert_size_stride(relu_46, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_47, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_142, (160, ), (1, ))
    assert_size_stride(relu_47, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_48, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_145, (192, ), (1, ))
    assert_size_stride(avg_pool2d_4, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_49, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_148, (192, ), (1, ))
    assert_size_stride(cat_5, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_50, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_151, (192, ), (1, ))
    assert_size_stride(convolution_51, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_154, (160, ), (1, ))
    assert_size_stride(relu_51, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_52, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_157, (160, ), (1, ))
    assert_size_stride(relu_52, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_53, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_160, (192, ), (1, ))
    assert_size_stride(convolution_54, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_163, (160, ), (1, ))
    assert_size_stride(relu_54, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_55, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_166, (160, ), (1, ))
    assert_size_stride(relu_55, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_56, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_169, (160, ), (1, ))
    assert_size_stride(relu_56, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_57, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_172, (160, ), (1, ))
    assert_size_stride(relu_57, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_58, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_175, (192, ), (1, ))
    assert_size_stride(avg_pool2d_5, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_59, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_178, (192, ), (1, ))
    assert_size_stride(cat_6, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_60, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_181, (192, ), (1, ))
    assert_size_stride(convolution_61, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_184, (192, ), (1, ))
    assert_size_stride(relu_61, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_62, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_187, (192, ), (1, ))
    assert_size_stride(relu_62, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_63, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_190, (192, ), (1, ))
    assert_size_stride(convolution_64, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_193, (192, ), (1, ))
    assert_size_stride(relu_64, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_65, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_196, (192, ), (1, ))
    assert_size_stride(relu_65, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_66, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_199, (192, ), (1, ))
    assert_size_stride(relu_66, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_67, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_202, (192, ), (1, ))
    assert_size_stride(relu_67, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_68, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_205, (192, ), (1, ))
    assert_size_stride(avg_pool2d_6, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_69, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_208, (192, ), (1, ))
    assert_size_stride(cat_7, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_70, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_211, (192, ), (1, ))
    assert_size_stride(relu_70, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_71, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_214, (320, ), (1, ))
    assert_size_stride(convolution_72, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_217, (192, ), (1, ))
    assert_size_stride(relu_72, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_73, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_220, (192, ), (1, ))
    assert_size_stride(relu_73, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_74, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_223, (192, ), (1, ))
    assert_size_stride(relu_74, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_75, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_226, (192, ), (1, ))
    assert_size_stride(getitem_159, (8, 768, 8, 8), (49152, 1, 6144, 768))
    assert_size_stride(cat_8, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(convolution_76, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_229, (320, ), (1, ))
    assert_size_stride(convolution_77, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_232, (384, ), (1, ))
    assert_size_stride(relu_77, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_78, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_235, (384, ), (1, ))
    assert_size_stride(convolution_79, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_238, (384, ), (1, ))
    assert_size_stride(convolution_80, (8, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(squeeze_241, (448, ), (1, ))
    assert_size_stride(relu_80, (8, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(convolution_81, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_244, (384, ), (1, ))
    assert_size_stride(relu_81, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_82, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_247, (384, ), (1, ))
    assert_size_stride(convolution_83, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_250, (384, ), (1, ))
    assert_size_stride(avg_pool2d_7, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(convolution_84, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_253, (192, ), (1, ))
    assert_size_stride(cat_11, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_85, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_256, (320, ), (1, ))
    assert_size_stride(convolution_86, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_259, (384, ), (1, ))
    assert_size_stride(relu_86, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_87, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_262, (384, ), (1, ))
    assert_size_stride(convolution_88, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_265, (384, ), (1, ))
    assert_size_stride(convolution_89, (8, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(squeeze_268, (448, ), (1, ))
    assert_size_stride(relu_89, (8, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(convolution_90, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_271, (384, ), (1, ))
    assert_size_stride(relu_90, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_91, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_274, (384, ), (1, ))
    assert_size_stride(convolution_92, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_277, (384, ), (1, ))
    assert_size_stride(avg_pool2d_8, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_93, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_280, (192, ), (1, ))
    assert_size_stride(clone, (8, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_378, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_1, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_390, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_2, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_402, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(le_5, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_438, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_6, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_450, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_8, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_474, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(le_9, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_486, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_10, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_498, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_11, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_510, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_522, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(le_14, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_546, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_15, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_558, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_17, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_582, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(le_18, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_594, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_630, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_22, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_642, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_24, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_666, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_25, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_678, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_702, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_30, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_738, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_33, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_774, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_34, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_786, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_35, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_798, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_810, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_822, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_40, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_858, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_870, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_882, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_43, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_894, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_44, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_906, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_45, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_918, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_930, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_942, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_954, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_966, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_50, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_978, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_990, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_1002, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_53, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1014, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_54, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1026, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_55, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1038, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1050, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1062, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1074, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1086, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_60, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1098, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1110, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1122, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_63, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1134, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_64, (8, 96, 17, 17), (27744, 1, 1632, 96))
    assert_size_stride(unsqueeze_1146, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1158, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1170, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_67, (8, 384, 17, 17), (110976, 1, 6528, 384))
    assert_size_stride(unsqueeze_1182, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_68, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1194, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_69, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1206, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1218, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1230, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_72, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1242, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1254, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_74, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1266, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_75, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1278, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_76, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1290, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1302, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1314, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_79, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1326, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1338, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_81, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1350, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_82, (8, 32, 35, 35), (39200, 1, 1120, 32))
    assert_size_stride(unsqueeze_1362, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_83, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1374, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1386, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1398, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_86, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1410, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1422, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_88, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1434, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1446, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1458, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_1470, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1482, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_1494, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((192, 4), (1, 192), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((192, 4), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_93, unsqueeze_378, buf3, buf5, 768, 128, grid=grid(768), stream=stream0)
        buf4 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 192, 4, grid=grid(192), stream=stream0)
        buf6 = empty((192, ), device='cuda', dtype=torch.float32)
        buf7 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(buf5, squeeze_280, buf6, buf7, 192, 4, grid=grid(192), stream=stream0)
        buf8 = empty_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4.run(le, buf0, convolution_93, unsqueeze_378, buf6, squeeze_280, buf4, primals_187, buf8, 98304, grid=grid(98304), stream=stream0)
        del convolution_93
        del le
        del primals_187
        del squeeze_280
        del unsqueeze_378
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf9 = aten.convolution_backward(buf8, avg_pool2d_8, primals_282, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_8
        del primals_282
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_5.run(buf10, buf12, 1048576, grid=grid(1048576), stream=stream0)
        del buf10
        buf13 = empty_strided((384, 4), (1, 384), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((384, 4), (1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_6.run(le_1, buf0, convolution_92, unsqueeze_390, buf13, buf15, 1536, 128, grid=grid(1536), stream=stream0)
        buf14 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf13, buf14, 384, 4, grid=grid(384), stream=stream0)
        buf16 = empty((384, ), device='cuda', dtype=torch.float32)
        buf17 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf15, squeeze_277, buf16, buf17, 384, 4, grid=grid(384), stream=stream0)
        buf18 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9.run(le_1, buf0, convolution_92, unsqueeze_390, buf16, squeeze_277, buf14, primals_185, buf18, 196608, grid=grid(196608), stream=stream0)
        del convolution_92
        del le_1
        del primals_185
        del squeeze_277
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf19 = aten.convolution_backward(buf18, relu_90, primals_281, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_281
        buf20 = buf19[0]
        buf21 = buf19[1]
        del buf19
        buf22 = buf15; del buf15  # reuse
        buf24 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_10.run(le_2, buf0, convolution_91, unsqueeze_402, buf22, buf24, 1536, 128, grid=grid(1536), stream=stream0)
        buf23 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf22, buf23, 384, 4, grid=grid(384), stream=stream0)
        buf25 = empty((384, ), device='cuda', dtype=torch.float32)
        buf26 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf24, squeeze_274, buf25, buf26, 384, 4, grid=grid(384), stream=stream0)
        buf27 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11.run(le_2, buf0, convolution_91, unsqueeze_402, buf25, squeeze_274, buf23, primals_183, buf27, 196608, grid=grid(196608), stream=stream0)
        del convolution_91
        del le_2
        del primals_183
        del squeeze_274
        del unsqueeze_402
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf28 = aten.convolution_backward(buf27, relu_90, primals_280, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_280
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_90, buf20, buf29, buf31, 384, 512, grid=grid(384), stream=stream0)
        buf32 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_90, buf20, buf29, convolution_90, unsqueeze_414, buf32, 1536, 128, grid=grid(1536), stream=stream0)
        buf33 = empty((384, ), device='cuda', dtype=torch.float32)
        buf35 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf32, squeeze_271, buf33, buf35, 384, 4, grid=grid(384), stream=stream0)
        buf34 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14.run(relu_90, buf20, buf29, convolution_90, unsqueeze_414, buf33, squeeze_271, buf31, primals_181, buf34, 512, 384, grid=grid(512, 384), stream=stream0)
        del buf20
        del buf29
        del convolution_90
        del primals_181
        del relu_90
        del squeeze_271
        del unsqueeze_414
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf36 = aten.convolution_backward(buf34, relu_89, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_279
        buf37 = buf36[0]
        buf38 = buf36[1]
        del buf36
        buf39 = empty_strided((448, 4), (1, 448), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((448, 4), (1, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_15.run(relu_89, buf37, convolution_89, unsqueeze_426, buf39, buf41, 1792, 128, grid=grid(1792), stream=stream0)
        buf40 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_16.run(buf39, buf40, 448, 4, grid=grid(448), stream=stream0)
        buf42 = empty((448, ), device='cuda', dtype=torch.float32)
        buf43 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf41, squeeze_268, buf42, buf43, 448, 4, grid=grid(448), stream=stream0)
        buf44 = empty_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(relu_89, buf37, convolution_89, unsqueeze_426, buf42, squeeze_268, buf40, primals_179, buf44, 512, 448, grid=grid(512, 448), stream=stream0)
        del buf37
        del convolution_89
        del primals_179
        del relu_89
        del squeeze_268
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf45 = aten.convolution_backward(buf44, cat_11, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_278
        buf46 = buf45[0]
        buf47 = buf45[1]
        del buf45
        buf48 = buf32; del buf32  # reuse
        buf50 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(le_5, buf0, convolution_88, unsqueeze_438, buf48, buf50, 1536, 128, grid=grid(1536), stream=stream0)
        buf49 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf48, buf49, 384, 4, grid=grid(384), stream=stream0)
        buf51 = empty((384, ), device='cuda', dtype=torch.float32)
        buf52 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf50, squeeze_265, buf51, buf52, 384, 4, grid=grid(384), stream=stream0)
        buf53 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20.run(le_5, buf0, convolution_88, unsqueeze_438, buf51, squeeze_265, buf49, primals_177, buf53, 196608, grid=grid(196608), stream=stream0)
        del convolution_88
        del le_5
        del primals_177
        del squeeze_265
        del unsqueeze_438
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf54 = aten.convolution_backward(buf53, relu_86, primals_277, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_277
        buf55 = buf54[0]
        buf56 = buf54[1]
        del buf54
        buf57 = buf50; del buf50  # reuse
        buf59 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_21.run(le_6, buf0, convolution_87, unsqueeze_450, buf57, buf59, 1536, 128, grid=grid(1536), stream=stream0)
        buf58 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf57, buf58, 384, 4, grid=grid(384), stream=stream0)
        del buf57
        buf60 = empty((384, ), device='cuda', dtype=torch.float32)
        buf61 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf59, squeeze_262, buf60, buf61, 384, 4, grid=grid(384), stream=stream0)
        buf62 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22.run(le_6, buf0, convolution_87, unsqueeze_450, buf60, squeeze_262, buf58, primals_175, buf62, 196608, grid=grid(196608), stream=stream0)
        del convolution_87
        del le_6
        del primals_175
        del squeeze_262
        del unsqueeze_450
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf63 = aten.convolution_backward(buf62, relu_86, primals_276, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_276
        buf64 = buf63[0]
        buf65 = buf63[1]
        del buf63
        buf66 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_86, buf55, buf64, buf66, 384, 512, grid=grid(384), stream=stream0)
        buf67 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_86, buf55, buf64, convolution_86, unsqueeze_462, buf67, 1536, 128, grid=grid(1536), stream=stream0)
        buf68 = empty((384, ), device='cuda', dtype=torch.float32)
        buf70 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf67, squeeze_259, buf68, buf70, 384, 4, grid=grid(384), stream=stream0)
        buf69 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14.run(relu_86, buf55, buf64, convolution_86, unsqueeze_462, buf68, squeeze_259, buf66, primals_173, buf69, 512, 384, grid=grid(512, 384), stream=stream0)
        del buf55
        del convolution_86
        del primals_173
        del relu_86
        del squeeze_259
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf71 = aten.convolution_backward(buf69, cat_11, primals_275, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_275
        buf72 = buf71[0]
        buf73 = buf71[1]
        del buf71
        buf74 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        buf76 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_23.run(le_8, buf0, convolution_85, unsqueeze_474, buf74, buf76, 1280, 128, grid=grid(1280), stream=stream0)
        buf75 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_24.run(buf74, buf75, 320, 4, grid=grid(320), stream=stream0)
        buf77 = empty((320, ), device='cuda', dtype=torch.float32)
        buf78 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_25.run(buf76, squeeze_256, buf77, buf78, 320, 4, grid=grid(320), stream=stream0)
        buf79 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(le_8, buf0, convolution_85, unsqueeze_474, buf77, squeeze_256, buf75, primals_171, buf79, 163840, grid=grid(163840), stream=stream0)
        del buf0
        del convolution_85
        del le_8
        del primals_171
        del squeeze_256
        del unsqueeze_474
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf80 = aten.convolution_backward(buf79, cat_11, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_11
        del primals_274
        buf81 = buf80[0]
        buf82 = buf80[1]
        del buf80
        buf83 = reinterpret_tensor(buf8, (8, 192, 8, 8), (12288, 64, 8, 1), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_27.run(le_9, buf12, buf46, buf72, buf81, buf83, 1536, 64, grid=grid(1536, 64), stream=stream0)
        del le_9
        buf84 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_28.run(buf83, buf84, 192, 512, grid=grid(192), stream=stream0)
        buf85 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_29.run(buf83, convolution_84, unsqueeze_486, buf85, 768, 128, grid=grid(768), stream=stream0)
        buf86 = empty((192, ), device='cuda', dtype=torch.float32)
        buf87 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(buf85, squeeze_253, buf86, buf87, 192, 4, grid=grid(192), stream=stream0)
        buf88 = empty_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_30.run(buf83, convolution_84, unsqueeze_486, buf86, squeeze_253, buf84, primals_169, buf88, 512, 192, grid=grid(512, 192), stream=stream0)
        del buf83
        del convolution_84
        del primals_169
        del squeeze_253
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf89 = aten.convolution_backward(buf88, avg_pool2d_7, primals_273, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_7
        del primals_273
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf93 = reinterpret_tensor(buf69, (8, 384, 8, 8), (24576, 64, 8, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_31.run(le_10, buf12, buf46, buf72, buf81, buf93, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del le_10
        buf94 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_32.run(buf93, buf94, 384, 512, grid=grid(384), stream=stream0)
        buf95 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf93, convolution_83, unsqueeze_498, buf95, 1536, 128, grid=grid(1536), stream=stream0)
        buf96 = empty((384, ), device='cuda', dtype=torch.float32)
        buf97 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf95, squeeze_250, buf96, buf97, 384, 4, grid=grid(384), stream=stream0)
        buf98 = reinterpret_tensor(buf64, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf93, convolution_83, unsqueeze_498, buf96, squeeze_250, buf94, primals_167, buf98, 512, 384, grid=grid(512, 384), stream=stream0)
        del convolution_83
        del primals_167
        del squeeze_250
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf99 = aten.convolution_backward(buf98, relu_81, primals_272, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_272
        buf100 = buf99[0]
        buf102 = reinterpret_tensor(buf98, (8, 384, 8, 8), (24576, 64, 8, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_35.run(le_11, buf12, buf46, buf72, buf81, buf102, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del le_11
        buf103 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_32.run(buf102, buf103, 384, 512, grid=grid(384), stream=stream0)
        buf104 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf102, convolution_82, unsqueeze_510, buf104, 1536, 128, grid=grid(1536), stream=stream0)
        buf105 = empty((384, ), device='cuda', dtype=torch.float32)
        buf106 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf104, squeeze_247, buf105, buf106, 384, 4, grid=grid(384), stream=stream0)
        buf107 = reinterpret_tensor(buf93, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf102, convolution_82, unsqueeze_510, buf105, squeeze_247, buf103, primals_165, buf107, 512, 384, grid=grid(512, 384), stream=stream0)
        del buf102
        del convolution_82
        del primals_165
        del squeeze_247
        del unsqueeze_510
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf108 = aten.convolution_backward(buf107, relu_81, primals_271, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_271
        buf109 = buf108[0]
        buf111 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_81, buf100, buf109, buf111, 384, 512, grid=grid(384), stream=stream0)
        buf112 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_81, buf100, buf109, convolution_81, unsqueeze_522, buf112, 1536, 128, grid=grid(1536), stream=stream0)
        buf113 = empty((384, ), device='cuda', dtype=torch.float32)
        buf115 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf112, squeeze_244, buf113, buf115, 384, 4, grid=grid(384), stream=stream0)
        buf114 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14.run(relu_81, buf100, buf109, convolution_81, unsqueeze_522, buf113, squeeze_244, buf111, primals_163, buf114, 512, 384, grid=grid(512, 384), stream=stream0)
        del buf100
        del convolution_81
        del primals_163
        del relu_81
        del squeeze_244
        del unsqueeze_522
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf116 = aten.convolution_backward(buf114, relu_80, primals_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_270
        buf117 = buf116[0]
        buf119 = buf41; del buf41  # reuse
        buf121 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_15.run(relu_80, buf117, convolution_80, unsqueeze_534, buf119, buf121, 1792, 128, grid=grid(1792), stream=stream0)
        buf120 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_16.run(buf119, buf120, 448, 4, grid=grid(448), stream=stream0)
        del buf119
        buf122 = empty((448, ), device='cuda', dtype=torch.float32)
        buf123 = empty((448, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf121, squeeze_241, buf122, buf123, 448, 4, grid=grid(448), stream=stream0)
        del buf121
        buf124 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18.run(relu_80, buf117, convolution_80, unsqueeze_534, buf122, squeeze_241, buf120, primals_161, buf124, 512, 448, grid=grid(512, 448), stream=stream0)
        del buf117
        del buf122
        del convolution_80
        del primals_161
        del relu_80
        del squeeze_241
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf125 = aten.convolution_backward(buf124, cat_8, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf124
        del primals_269
        buf126 = buf125[0]
        buf128 = reinterpret_tensor(buf114, (8, 384, 8, 8), (24576, 64, 8, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_36.run(le_14, buf12, buf46, buf72, buf81, buf128, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del le_14
        buf129 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_32.run(buf128, buf129, 384, 512, grid=grid(384), stream=stream0)
        buf130 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf128, convolution_79, unsqueeze_546, buf130, 1536, 128, grid=grid(1536), stream=stream0)
        buf131 = empty((384, ), device='cuda', dtype=torch.float32)
        buf132 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf130, squeeze_238, buf131, buf132, 384, 4, grid=grid(384), stream=stream0)
        buf133 = reinterpret_tensor(buf109, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf128, convolution_79, unsqueeze_546, buf131, squeeze_238, buf129, primals_159, buf133, 512, 384, grid=grid(512, 384), stream=stream0)
        del convolution_79
        del primals_159
        del squeeze_238
        del unsqueeze_546
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf134 = aten.convolution_backward(buf133, relu_77, primals_268, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_268
        buf135 = buf134[0]
        buf137 = reinterpret_tensor(buf133, (8, 384, 8, 8), (24576, 64, 8, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_37.run(le_15, buf12, buf46, buf72, buf81, buf137, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del le_15
        buf138 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_32.run(buf137, buf138, 384, 512, grid=grid(384), stream=stream0)
        buf139 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf137, convolution_78, unsqueeze_558, buf139, 1536, 128, grid=grid(1536), stream=stream0)
        buf140 = empty((384, ), device='cuda', dtype=torch.float32)
        buf141 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf139, squeeze_235, buf140, buf141, 384, 4, grid=grid(384), stream=stream0)
        buf142 = reinterpret_tensor(buf128, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf137, convolution_78, unsqueeze_558, buf140, squeeze_235, buf138, primals_157, buf142, 512, 384, grid=grid(512, 384), stream=stream0)
        del buf137
        del convolution_78
        del primals_157
        del squeeze_235
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf143 = aten.convolution_backward(buf142, relu_77, primals_267, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_267
        buf144 = buf143[0]
        buf146 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_12.run(relu_77, buf135, buf144, buf146, 384, 512, grid=grid(384), stream=stream0)
        buf147 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_77, buf135, buf144, convolution_77, unsqueeze_570, buf147, 1536, 128, grid=grid(1536), stream=stream0)
        buf148 = empty((384, ), device='cuda', dtype=torch.float32)
        buf150 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf147, squeeze_232, buf148, buf150, 384, 4, grid=grid(384), stream=stream0)
        del buf147
        buf149 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14.run(relu_77, buf135, buf144, convolution_77, unsqueeze_570, buf148, squeeze_232, buf146, primals_155, buf149, 512, 384, grid=grid(512, 384), stream=stream0)
        del buf135
        del buf144
        del convolution_77
        del primals_155
        del relu_77
        del squeeze_232
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf151 = aten.convolution_backward(buf149, cat_8, primals_266, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf149
        del primals_266
        buf152 = buf151[0]
        buf154 = reinterpret_tensor(buf79, (8, 320, 8, 8), (20480, 64, 8, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_38.run(le_17, buf12, buf46, buf72, buf81, buf154, 2560, 64, grid=grid(2560, 64), stream=stream0)
        del buf12
        del buf46
        del buf72
        del buf81
        del le_17
        buf155 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_39.run(buf154, buf155, 320, 512, grid=grid(320), stream=stream0)
        buf156 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_40.run(buf154, convolution_76, unsqueeze_582, buf156, 1280, 128, grid=grid(1280), stream=stream0)
        buf157 = empty((320, ), device='cuda', dtype=torch.float32)
        buf158 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_25.run(buf156, squeeze_229, buf157, buf158, 320, 4, grid=grid(320), stream=stream0)
        buf159 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_41.run(buf154, convolution_76, unsqueeze_582, buf157, squeeze_229, buf155, primals_153, buf159, 512, 320, grid=grid(512, 320), stream=stream0)
        del buf154
        del convolution_76
        del primals_153
        del squeeze_229
        del unsqueeze_582
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf160 = aten.convolution_backward(buf159, cat_8, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_8
        del primals_265
        buf161 = buf160[0]
        buf163 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward]
        triton_poi_fused_add_avg_pool2d_backward_42.run(buf163, buf90, buf152, buf161, 655360, grid=grid(655360), stream=stream0)
        del buf152
        del buf161
        del buf90
        buf101 = buf99[1]
        del buf99
        buf110 = buf108[1]
        del buf108
        buf118 = buf116[1]
        del buf116
        buf127 = buf125[1]
        del buf125
        buf136 = buf134[1]
        del buf134
        buf145 = buf143[1]
        del buf143
        buf153 = buf151[1]
        del buf151
        buf162 = buf160[1]
        del buf160
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf164 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf163, (8, 768, 8, 8), (81920, 64, 8, 1), 32768), cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_159)
        del getitem_159
        buf165 = buf164
        del buf164
        buf166 = buf85; del buf85  # reuse
        buf168 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(le_18, buf163, convolution_75, unsqueeze_594, buf166, buf168, 768, 128, grid=grid(768), stream=stream0)
        buf167 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_2.run(buf166, buf167, 192, 4, grid=grid(192), stream=stream0)
        del buf166
        buf169 = empty((192, ), device='cuda', dtype=torch.float32)
        buf170 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_3.run(buf168, squeeze_226, buf169, buf170, 192, 4, grid=grid(192), stream=stream0)
        del buf168
        buf171 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44.run(le_18, buf163, convolution_75, unsqueeze_594, buf169, squeeze_226, buf167, primals_151, buf171, 512, 192, grid=grid(512, 192), stream=stream0)
        del convolution_75
        del le_18
        del primals_151
        del squeeze_226
        del unsqueeze_594
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf172 = aten.convolution_backward(buf171, relu_74, primals_264, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf171
        del primals_264
        buf173 = buf172[0]
        buf174 = buf172[1]
        del buf172
        buf175 = empty((192, 19), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_74, buf173, buf175, 3648, 122, grid=grid(3648), stream=stream0)
        buf176 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf175, buf176, 192, 19, grid=grid(192), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (192, 19), (1, 192), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_74, buf173, convolution_74, unsqueeze_606, buf177, 3648, 122, grid=grid(3648), stream=stream0)
        buf178 = empty((192, ), device='cuda', dtype=torch.float32)
        buf179 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf177, squeeze_223, buf178, buf179, 192, 19, grid=grid(192), stream=stream0)
        buf180 = empty_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_74, buf173, convolution_74, unsqueeze_606, buf178, squeeze_223, buf176, primals_149, buf180, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf173
        del convolution_74
        del primals_149
        del relu_74
        del squeeze_223
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf181 = aten.convolution_backward(buf180, relu_73, primals_263, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_263
        buf182 = buf181[0]
        buf183 = buf181[1]
        del buf181
        buf184 = reinterpret_tensor(buf177, (192, 19), (19, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_73, buf182, buf184, 3648, 122, grid=grid(3648), stream=stream0)
        buf185 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf184, buf185, 192, 19, grid=grid(192), stream=stream0)
        buf186 = reinterpret_tensor(buf184, (192, 19), (1, 192), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_73, buf182, convolution_73, unsqueeze_618, buf186, 3648, 122, grid=grid(3648), stream=stream0)
        buf187 = empty((192, ), device='cuda', dtype=torch.float32)
        buf188 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf186, squeeze_220, buf187, buf188, 192, 19, grid=grid(192), stream=stream0)
        buf189 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_73, buf182, convolution_73, unsqueeze_618, buf187, squeeze_220, buf185, primals_147, buf189, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf182
        del convolution_73
        del primals_147
        del relu_73
        del squeeze_220
        del unsqueeze_618
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf190 = aten.convolution_backward(buf189, relu_72, primals_262, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_262
        buf191 = buf190[0]
        buf192 = buf190[1]
        del buf190
        buf193 = reinterpret_tensor(buf186, (192, 19), (19, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_72, buf191, buf193, 3648, 122, grid=grid(3648), stream=stream0)
        buf194 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf193, buf194, 192, 19, grid=grid(192), stream=stream0)
        buf195 = reinterpret_tensor(buf193, (192, 19), (1, 192), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_72, buf191, convolution_72, unsqueeze_630, buf195, 3648, 122, grid=grid(3648), stream=stream0)
        buf196 = empty((192, ), device='cuda', dtype=torch.float32)
        buf197 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf195, squeeze_217, buf196, buf197, 192, 19, grid=grid(192), stream=stream0)
        buf198 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_72, buf191, convolution_72, unsqueeze_630, buf196, squeeze_217, buf194, primals_145, buf198, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf191
        del convolution_72
        del primals_145
        del relu_72
        del squeeze_217
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf199 = aten.convolution_backward(buf198, cat_7, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_261
        buf200 = buf199[0]
        buf201 = buf199[1]
        del buf199
        buf202 = buf156; del buf156  # reuse
        buf204 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_50.run(le_22, buf163, convolution_71, unsqueeze_642, buf202, buf204, 1280, 128, grid=grid(1280), stream=stream0)
        buf203 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_24.run(buf202, buf203, 320, 4, grid=grid(320), stream=stream0)
        del buf202
        buf205 = empty((320, ), device='cuda', dtype=torch.float32)
        buf206 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_25.run(buf204, squeeze_214, buf205, buf206, 320, 4, grid=grid(320), stream=stream0)
        del buf204
        buf207 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51.run(le_22, buf163, convolution_71, unsqueeze_642, buf205, squeeze_214, buf203, primals_143, buf207, 512, 320, grid=grid(512, 320), stream=stream0)
        del buf163
        del buf205
        del convolution_71
        del le_22
        del primals_143
        del squeeze_214
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf208 = aten.convolution_backward(buf207, relu_70, primals_260, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf207
        del primals_260
        buf209 = buf208[0]
        buf210 = buf208[1]
        del buf208
        buf211 = reinterpret_tensor(buf195, (192, 19), (19, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_70, buf209, buf211, 3648, 122, grid=grid(3648), stream=stream0)
        buf212 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf211, buf212, 192, 19, grid=grid(192), stream=stream0)
        buf213 = reinterpret_tensor(buf211, (192, 19), (1, 192), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_70, buf209, convolution_70, unsqueeze_654, buf213, 3648, 122, grid=grid(3648), stream=stream0)
        buf214 = empty((192, ), device='cuda', dtype=torch.float32)
        buf215 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf213, squeeze_211, buf214, buf215, 192, 19, grid=grid(192), stream=stream0)
        buf216 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_70, buf209, convolution_70, unsqueeze_654, buf214, squeeze_211, buf212, primals_141, buf216, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf209
        del convolution_70
        del primals_141
        del relu_70
        del squeeze_211
        del unsqueeze_654
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf217 = aten.convolution_backward(buf216, cat_7, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_7
        del primals_259
        buf218 = buf217[0]
        buf219 = buf217[1]
        del buf217
        buf220 = reinterpret_tensor(buf213, (192, 19), (19, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_52.run(le_24, buf165, buf200, buf218, buf220, 3648, 122, grid=grid(3648), stream=stream0)
        buf221 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf220, buf221, 192, 19, grid=grid(192), stream=stream0)
        buf222 = reinterpret_tensor(buf220, (192, 19), (1, 192), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(le_24, buf165, buf200, buf218, convolution_69, unsqueeze_666, buf222, 3648, 122, grid=grid(3648), stream=stream0)
        buf223 = empty((192, ), device='cuda', dtype=torch.float32)
        buf225 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf222, squeeze_208, buf223, buf225, 192, 19, grid=grid(192), stream=stream0)
        buf224 = buf216; del buf216  # reuse
        buf226 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54.run(buf226, le_24, buf165, buf200, buf218, convolution_69, unsqueeze_666, buf223, squeeze_208, buf221, primals_139, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_69
        del le_24
        del primals_139
        del squeeze_208
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf227 = aten.convolution_backward(buf226, avg_pool2d_6, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_6
        del primals_258
        buf228 = buf227[0]
        buf229 = buf227[1]
        del buf227
        buf230 = empty((8, 768, 17, 17), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_55.run(buf228, buf230, 1775616, grid=grid(1775616), stream=stream0)
        del buf228
        buf231 = reinterpret_tensor(buf222, (192, 19), (19, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_56.run(le_25, buf165, buf200, buf218, buf231, 3648, 122, grid=grid(3648), stream=stream0)
        buf232 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf231, buf232, 192, 19, grid=grid(192), stream=stream0)
        buf233 = reinterpret_tensor(buf231, (192, 19), (1, 192), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(le_25, buf165, buf200, buf218, convolution_68, unsqueeze_678, buf233, 3648, 122, grid=grid(3648), stream=stream0)
        buf234 = empty((192, ), device='cuda', dtype=torch.float32)
        buf236 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf233, squeeze_205, buf234, buf236, 192, 19, grid=grid(192), stream=stream0)
        buf235 = buf226; del buf226  # reuse
        buf237 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58.run(buf237, le_25, buf165, buf200, buf218, convolution_68, unsqueeze_678, buf234, squeeze_205, buf232, primals_137, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_68
        del le_25
        del primals_137
        del squeeze_205
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf238 = aten.convolution_backward(buf237, relu_67, primals_257, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_257
        buf239 = buf238[0]
        buf240 = buf238[1]
        del buf238
        buf241 = reinterpret_tensor(buf233, (192, 19), (19, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_67, buf239, buf241, 3648, 122, grid=grid(3648), stream=stream0)
        buf242 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf241, buf242, 192, 19, grid=grid(192), stream=stream0)
        buf243 = reinterpret_tensor(buf241, (192, 19), (1, 192), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_67, buf239, convolution_67, unsqueeze_690, buf243, 3648, 122, grid=grid(3648), stream=stream0)
        buf244 = empty((192, ), device='cuda', dtype=torch.float32)
        buf245 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf243, squeeze_202, buf244, buf245, 192, 19, grid=grid(192), stream=stream0)
        buf246 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_67, buf239, convolution_67, unsqueeze_690, buf244, squeeze_202, buf242, primals_135, buf246, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf239
        del convolution_67
        del primals_135
        del relu_67
        del squeeze_202
        del unsqueeze_690
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf247 = aten.convolution_backward(buf246, relu_66, primals_256, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_256
        buf248 = buf247[0]
        buf249 = buf247[1]
        del buf247
        buf250 = reinterpret_tensor(buf243, (192, 19), (19, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_66, buf248, buf250, 3648, 122, grid=grid(3648), stream=stream0)
        buf251 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf250, buf251, 192, 19, grid=grid(192), stream=stream0)
        buf252 = reinterpret_tensor(buf250, (192, 19), (1, 192), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_66, buf248, convolution_66, unsqueeze_702, buf252, 3648, 122, grid=grid(3648), stream=stream0)
        buf253 = empty((192, ), device='cuda', dtype=torch.float32)
        buf254 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf252, squeeze_199, buf253, buf254, 192, 19, grid=grid(192), stream=stream0)
        buf255 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_66, buf248, convolution_66, unsqueeze_702, buf253, squeeze_199, buf251, primals_133, buf255, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf248
        del convolution_66
        del primals_133
        del relu_66
        del squeeze_199
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf256 = aten.convolution_backward(buf255, relu_65, primals_255, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_255
        buf257 = buf256[0]
        buf258 = buf256[1]
        del buf256
        buf259 = reinterpret_tensor(buf252, (192, 19), (19, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_65, buf257, buf259, 3648, 122, grid=grid(3648), stream=stream0)
        buf260 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf259, buf260, 192, 19, grid=grid(192), stream=stream0)
        buf261 = reinterpret_tensor(buf259, (192, 19), (1, 192), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_65, buf257, convolution_65, unsqueeze_714, buf261, 3648, 122, grid=grid(3648), stream=stream0)
        buf262 = empty((192, ), device='cuda', dtype=torch.float32)
        buf263 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf261, squeeze_196, buf262, buf263, 192, 19, grid=grid(192), stream=stream0)
        buf264 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_65, buf257, convolution_65, unsqueeze_714, buf262, squeeze_196, buf260, primals_131, buf264, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf257
        del convolution_65
        del primals_131
        del relu_65
        del squeeze_196
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf265 = aten.convolution_backward(buf264, relu_64, primals_254, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_254
        buf266 = buf265[0]
        buf267 = buf265[1]
        del buf265
        buf268 = reinterpret_tensor(buf261, (192, 19), (19, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_64, buf266, buf268, 3648, 122, grid=grid(3648), stream=stream0)
        buf269 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf268, buf269, 192, 19, grid=grid(192), stream=stream0)
        buf270 = reinterpret_tensor(buf268, (192, 19), (1, 192), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_64, buf266, convolution_64, unsqueeze_726, buf270, 3648, 122, grid=grid(3648), stream=stream0)
        buf271 = empty((192, ), device='cuda', dtype=torch.float32)
        buf272 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf270, squeeze_193, buf271, buf272, 192, 19, grid=grid(192), stream=stream0)
        buf273 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_64, buf266, convolution_64, unsqueeze_726, buf271, squeeze_193, buf269, primals_129, buf273, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf266
        del convolution_64
        del primals_129
        del relu_64
        del squeeze_193
        del unsqueeze_726
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf274 = aten.convolution_backward(buf273, cat_6, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_253
        buf275 = buf274[0]
        buf276 = buf274[1]
        del buf274
        buf277 = reinterpret_tensor(buf270, (192, 19), (19, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(le_30, buf165, buf200, buf218, buf277, 3648, 122, grid=grid(3648), stream=stream0)
        buf278 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf277, buf278, 192, 19, grid=grid(192), stream=stream0)
        buf279 = reinterpret_tensor(buf277, (192, 19), (1, 192), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_60.run(le_30, buf165, buf200, buf218, convolution_63, unsqueeze_738, buf279, 3648, 122, grid=grid(3648), stream=stream0)
        buf280 = empty((192, ), device='cuda', dtype=torch.float32)
        buf282 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf279, squeeze_190, buf280, buf282, 192, 19, grid=grid(192), stream=stream0)
        buf281 = buf273; del buf273  # reuse
        buf283 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61.run(buf283, le_30, buf165, buf200, buf218, convolution_63, unsqueeze_738, buf280, squeeze_190, buf278, primals_127, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_63
        del le_30
        del primals_127
        del squeeze_190
        del unsqueeze_738
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf284 = aten.convolution_backward(buf283, relu_62, primals_252, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_252
        buf285 = buf284[0]
        buf286 = buf284[1]
        del buf284
        buf287 = reinterpret_tensor(buf279, (192, 19), (19, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_62, buf285, buf287, 3648, 122, grid=grid(3648), stream=stream0)
        buf288 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf287, buf288, 192, 19, grid=grid(192), stream=stream0)
        buf289 = reinterpret_tensor(buf287, (192, 19), (1, 192), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_62, buf285, convolution_62, unsqueeze_750, buf289, 3648, 122, grid=grid(3648), stream=stream0)
        buf290 = empty((192, ), device='cuda', dtype=torch.float32)
        buf291 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf289, squeeze_187, buf290, buf291, 192, 19, grid=grid(192), stream=stream0)
        buf292 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_62, buf285, convolution_62, unsqueeze_750, buf290, squeeze_187, buf288, primals_125, buf292, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf285
        del convolution_62
        del primals_125
        del relu_62
        del squeeze_187
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf293 = aten.convolution_backward(buf292, relu_61, primals_251, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_251
        buf294 = buf293[0]
        buf295 = buf293[1]
        del buf293
        buf296 = reinterpret_tensor(buf289, (192, 19), (19, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_45.run(relu_61, buf294, buf296, 3648, 122, grid=grid(3648), stream=stream0)
        buf297 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf296, buf297, 192, 19, grid=grid(192), stream=stream0)
        buf298 = reinterpret_tensor(buf296, (192, 19), (1, 192), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(relu_61, buf294, convolution_61, unsqueeze_762, buf298, 3648, 122, grid=grid(3648), stream=stream0)
        buf299 = empty((192, ), device='cuda', dtype=torch.float32)
        buf300 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf298, squeeze_184, buf299, buf300, 192, 19, grid=grid(192), stream=stream0)
        buf301 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(relu_61, buf294, convolution_61, unsqueeze_762, buf299, squeeze_184, buf297, primals_123, buf301, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_61
        del primals_123
        del relu_61
        del squeeze_184
        del unsqueeze_762
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf302 = aten.convolution_backward(buf301, cat_6, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_250
        buf303 = buf302[0]
        buf304 = buf302[1]
        del buf302
        buf305 = reinterpret_tensor(buf298, (192, 19), (19, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_62.run(le_33, buf165, buf200, buf218, buf305, 3648, 122, grid=grid(3648), stream=stream0)
        buf306 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(buf305, buf306, 192, 19, grid=grid(192), stream=stream0)
        buf307 = reinterpret_tensor(buf305, (192, 19), (1, 192), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_63.run(le_33, buf165, buf200, buf218, convolution_60, unsqueeze_774, buf307, 3648, 122, grid=grid(3648), stream=stream0)
        buf308 = empty((192, ), device='cuda', dtype=torch.float32)
        buf310 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_48.run(buf307, squeeze_181, buf308, buf310, 192, 19, grid=grid(192), stream=stream0)
        buf309 = buf301; del buf301  # reuse
        buf311 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64.run(buf311, le_33, buf165, buf200, buf218, convolution_60, unsqueeze_774, buf308, squeeze_181, buf306, primals_121, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf165
        del buf200
        del convolution_60
        del le_33
        del primals_121
        del squeeze_181
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf312 = aten.convolution_backward(buf311, cat_6, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_6
        del primals_249
        buf313 = buf312[0]
        buf314 = buf312[1]
        del buf312
        buf315 = reinterpret_tensor(buf311, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_65.run(le_34, buf230, buf275, buf303, buf313, buf315, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_34
        buf316 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf315, buf316, 192, 2312, grid=grid(192), stream=stream0)
        buf317 = reinterpret_tensor(buf307, (192, 19), (19, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf315, convolution_59, unsqueeze_786, buf317, 3648, 122, grid=grid(3648), stream=stream0)
        buf318 = empty((192, ), device='cuda', dtype=torch.float32)
        buf319 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf317, squeeze_178, buf318, buf319, 192, 19, grid=grid(192), stream=stream0)
        buf320 = reinterpret_tensor(buf294, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf315, convolution_59, unsqueeze_786, buf318, squeeze_178, buf316, primals_119, buf320, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_59
        del primals_119
        del squeeze_178
        del unsqueeze_786
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf321 = aten.convolution_backward(buf320, avg_pool2d_5, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_5
        del primals_248
        buf322 = buf321[0]
        buf323 = buf321[1]
        del buf321
        buf324 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_55.run(buf322, buf324, 1775616, grid=grid(1775616), stream=stream0)
        del buf322
        buf325 = reinterpret_tensor(buf320, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_70.run(le_35, buf230, buf275, buf303, buf313, buf325, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_35
        buf326 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf325, buf326, 192, 2312, grid=grid(192), stream=stream0)
        buf327 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf325, convolution_58, unsqueeze_798, buf327, 3648, 122, grid=grid(3648), stream=stream0)
        buf328 = empty((192, ), device='cuda', dtype=torch.float32)
        buf329 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf327, squeeze_175, buf328, buf329, 192, 19, grid=grid(192), stream=stream0)
        buf330 = reinterpret_tensor(buf315, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf325, convolution_58, unsqueeze_798, buf328, squeeze_175, buf326, primals_117, buf330, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_58
        del primals_117
        del squeeze_175
        del unsqueeze_798
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf331 = aten.convolution_backward(buf330, relu_57, primals_247, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_247
        buf332 = buf331[0]
        buf333 = buf331[1]
        del buf331
        buf334 = empty((160, 19), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_57, buf332, buf334, 3040, 122, grid=grid(3040), stream=stream0)
        buf335 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf334, buf335, 160, 19, grid=grid(160), stream=stream0)
        buf336 = reinterpret_tensor(buf334, (160, 19), (1, 160), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_57, buf332, convolution_57, unsqueeze_810, buf336, 3040, 122, grid=grid(3040), stream=stream0)
        buf337 = empty((160, ), device='cuda', dtype=torch.float32)
        buf338 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf336, squeeze_172, buf337, buf338, 160, 19, grid=grid(160), stream=stream0)
        buf339 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_57, buf332, convolution_57, unsqueeze_810, buf337, squeeze_172, buf335, primals_115, buf339, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf332
        del convolution_57
        del primals_115
        del relu_57
        del squeeze_172
        del unsqueeze_810
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf340 = aten.convolution_backward(buf339, relu_56, primals_246, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_246
        buf341 = buf340[0]
        buf342 = buf340[1]
        del buf340
        buf343 = reinterpret_tensor(buf336, (160, 19), (19, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_56, buf341, buf343, 3040, 122, grid=grid(3040), stream=stream0)
        buf344 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf343, buf344, 160, 19, grid=grid(160), stream=stream0)
        buf345 = reinterpret_tensor(buf343, (160, 19), (1, 160), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_56, buf341, convolution_56, unsqueeze_822, buf345, 3040, 122, grid=grid(3040), stream=stream0)
        buf346 = empty((160, ), device='cuda', dtype=torch.float32)
        buf347 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf345, squeeze_169, buf346, buf347, 160, 19, grid=grid(160), stream=stream0)
        buf348 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_56, buf341, convolution_56, unsqueeze_822, buf346, squeeze_169, buf344, primals_113, buf348, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf341
        del convolution_56
        del primals_113
        del relu_56
        del squeeze_169
        del unsqueeze_822
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf349 = aten.convolution_backward(buf348, relu_55, primals_245, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_245
        buf350 = buf349[0]
        buf351 = buf349[1]
        del buf349
        buf352 = reinterpret_tensor(buf345, (160, 19), (19, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_55, buf350, buf352, 3040, 122, grid=grid(3040), stream=stream0)
        buf353 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf352, buf353, 160, 19, grid=grid(160), stream=stream0)
        buf354 = reinterpret_tensor(buf352, (160, 19), (1, 160), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_55, buf350, convolution_55, unsqueeze_834, buf354, 3040, 122, grid=grid(3040), stream=stream0)
        buf355 = empty((160, ), device='cuda', dtype=torch.float32)
        buf356 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf354, squeeze_166, buf355, buf356, 160, 19, grid=grid(160), stream=stream0)
        buf357 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_55, buf350, convolution_55, unsqueeze_834, buf355, squeeze_166, buf353, primals_111, buf357, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf350
        del convolution_55
        del primals_111
        del relu_55
        del squeeze_166
        del unsqueeze_834
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf358 = aten.convolution_backward(buf357, relu_54, primals_244, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_244
        buf359 = buf358[0]
        buf360 = buf358[1]
        del buf358
        buf361 = reinterpret_tensor(buf354, (160, 19), (19, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_54, buf359, buf361, 3040, 122, grid=grid(3040), stream=stream0)
        buf362 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf361, buf362, 160, 19, grid=grid(160), stream=stream0)
        buf363 = reinterpret_tensor(buf361, (160, 19), (1, 160), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_54, buf359, convolution_54, unsqueeze_846, buf363, 3040, 122, grid=grid(3040), stream=stream0)
        buf364 = empty((160, ), device='cuda', dtype=torch.float32)
        buf365 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf363, squeeze_163, buf364, buf365, 160, 19, grid=grid(160), stream=stream0)
        buf366 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_54, buf359, convolution_54, unsqueeze_846, buf364, squeeze_163, buf362, primals_109, buf366, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf359
        del convolution_54
        del primals_109
        del relu_54
        del squeeze_163
        del unsqueeze_846
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf367 = aten.convolution_backward(buf366, cat_5, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_243
        buf368 = buf367[0]
        buf369 = buf367[1]
        del buf367
        buf370 = reinterpret_tensor(buf330, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_76.run(le_40, buf230, buf275, buf303, buf313, buf370, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_40
        buf371 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf370, buf371, 192, 2312, grid=grid(192), stream=stream0)
        buf372 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf370, convolution_53, unsqueeze_858, buf372, 3648, 122, grid=grid(3648), stream=stream0)
        buf373 = empty((192, ), device='cuda', dtype=torch.float32)
        buf374 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf372, squeeze_160, buf373, buf374, 192, 19, grid=grid(192), stream=stream0)
        buf375 = reinterpret_tensor(buf325, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf370, convolution_53, unsqueeze_858, buf373, squeeze_160, buf371, primals_107, buf375, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_53
        del primals_107
        del squeeze_160
        del unsqueeze_858
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf376 = aten.convolution_backward(buf375, relu_52, primals_242, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_242
        buf377 = buf376[0]
        buf378 = buf376[1]
        del buf376
        buf379 = reinterpret_tensor(buf363, (160, 19), (19, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_52, buf377, buf379, 3040, 122, grid=grid(3040), stream=stream0)
        buf380 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf379, buf380, 160, 19, grid=grid(160), stream=stream0)
        buf381 = reinterpret_tensor(buf379, (160, 19), (1, 160), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_52, buf377, convolution_52, unsqueeze_870, buf381, 3040, 122, grid=grid(3040), stream=stream0)
        buf382 = empty((160, ), device='cuda', dtype=torch.float32)
        buf383 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf381, squeeze_157, buf382, buf383, 160, 19, grid=grid(160), stream=stream0)
        buf384 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_52, buf377, convolution_52, unsqueeze_870, buf382, squeeze_157, buf380, primals_105, buf384, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf377
        del convolution_52
        del primals_105
        del relu_52
        del squeeze_157
        del unsqueeze_870
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf385 = aten.convolution_backward(buf384, relu_51, primals_241, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_241
        buf386 = buf385[0]
        buf387 = buf385[1]
        del buf385
        buf388 = reinterpret_tensor(buf381, (160, 19), (19, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_51, buf386, buf388, 3040, 122, grid=grid(3040), stream=stream0)
        buf389 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf388, buf389, 160, 19, grid=grid(160), stream=stream0)
        buf390 = reinterpret_tensor(buf388, (160, 19), (1, 160), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_51, buf386, convolution_51, unsqueeze_882, buf390, 3040, 122, grid=grid(3040), stream=stream0)
        buf391 = empty((160, ), device='cuda', dtype=torch.float32)
        buf392 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf390, squeeze_154, buf391, buf392, 160, 19, grid=grid(160), stream=stream0)
        buf393 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_51, buf386, convolution_51, unsqueeze_882, buf391, squeeze_154, buf389, primals_103, buf393, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf386
        del convolution_51
        del primals_103
        del relu_51
        del squeeze_154
        del unsqueeze_882
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf394 = aten.convolution_backward(buf393, cat_5, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_240
        buf395 = buf394[0]
        buf396 = buf394[1]
        del buf394
        buf397 = reinterpret_tensor(buf375, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_77.run(le_43, buf230, buf275, buf303, buf313, buf397, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del buf230
        del buf275
        del buf303
        del le_43
        buf398 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf397, buf398, 192, 2312, grid=grid(192), stream=stream0)
        buf399 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf397, convolution_50, unsqueeze_894, buf399, 3648, 122, grid=grid(3648), stream=stream0)
        buf400 = empty((192, ), device='cuda', dtype=torch.float32)
        buf401 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf399, squeeze_151, buf400, buf401, 192, 19, grid=grid(192), stream=stream0)
        buf402 = reinterpret_tensor(buf370, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf397, convolution_50, unsqueeze_894, buf400, squeeze_151, buf398, primals_101, buf402, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_50
        del primals_101
        del squeeze_151
        del unsqueeze_894
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf403 = aten.convolution_backward(buf402, cat_5, primals_239, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_5
        del primals_239
        buf404 = buf403[0]
        buf405 = buf403[1]
        del buf403
        buf406 = reinterpret_tensor(buf402, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_65.run(le_44, buf324, buf368, buf395, buf404, buf406, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_44
        buf407 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf406, buf407, 192, 2312, grid=grid(192), stream=stream0)
        buf408 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf406, convolution_49, unsqueeze_906, buf408, 3648, 122, grid=grid(3648), stream=stream0)
        buf409 = empty((192, ), device='cuda', dtype=torch.float32)
        buf410 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf408, squeeze_148, buf409, buf410, 192, 19, grid=grid(192), stream=stream0)
        buf411 = reinterpret_tensor(buf397, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf406, convolution_49, unsqueeze_906, buf409, squeeze_148, buf407, primals_99, buf411, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_49
        del primals_99
        del squeeze_148
        del unsqueeze_906
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf412 = aten.convolution_backward(buf411, avg_pool2d_4, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_4
        del primals_238
        buf413 = buf412[0]
        buf414 = buf412[1]
        del buf412
        buf415 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_55.run(buf413, buf415, 1775616, grid=grid(1775616), stream=stream0)
        del buf413
        buf416 = reinterpret_tensor(buf411, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_70.run(le_45, buf324, buf368, buf395, buf404, buf416, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_45
        buf417 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf416, buf417, 192, 2312, grid=grid(192), stream=stream0)
        buf418 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf416, convolution_48, unsqueeze_918, buf418, 3648, 122, grid=grid(3648), stream=stream0)
        buf419 = empty((192, ), device='cuda', dtype=torch.float32)
        buf420 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf418, squeeze_145, buf419, buf420, 192, 19, grid=grid(192), stream=stream0)
        buf421 = reinterpret_tensor(buf406, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf416, convolution_48, unsqueeze_918, buf419, squeeze_145, buf417, primals_97, buf421, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_48
        del primals_97
        del squeeze_145
        del unsqueeze_918
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf422 = aten.convolution_backward(buf421, relu_47, primals_237, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_237
        buf423 = buf422[0]
        buf424 = buf422[1]
        del buf422
        buf425 = reinterpret_tensor(buf390, (160, 19), (19, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_47, buf423, buf425, 3040, 122, grid=grid(3040), stream=stream0)
        buf426 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf425, buf426, 160, 19, grid=grid(160), stream=stream0)
        buf427 = reinterpret_tensor(buf425, (160, 19), (1, 160), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_47, buf423, convolution_47, unsqueeze_930, buf427, 3040, 122, grid=grid(3040), stream=stream0)
        buf428 = empty((160, ), device='cuda', dtype=torch.float32)
        buf429 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf427, squeeze_142, buf428, buf429, 160, 19, grid=grid(160), stream=stream0)
        buf430 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_47, buf423, convolution_47, unsqueeze_930, buf428, squeeze_142, buf426, primals_95, buf430, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf423
        del convolution_47
        del primals_95
        del relu_47
        del squeeze_142
        del unsqueeze_930
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf431 = aten.convolution_backward(buf430, relu_46, primals_236, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_236
        buf432 = buf431[0]
        buf433 = buf431[1]
        del buf431
        buf434 = reinterpret_tensor(buf427, (160, 19), (19, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_46, buf432, buf434, 3040, 122, grid=grid(3040), stream=stream0)
        buf435 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf434, buf435, 160, 19, grid=grid(160), stream=stream0)
        buf436 = reinterpret_tensor(buf434, (160, 19), (1, 160), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_46, buf432, convolution_46, unsqueeze_942, buf436, 3040, 122, grid=grid(3040), stream=stream0)
        buf437 = empty((160, ), device='cuda', dtype=torch.float32)
        buf438 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf436, squeeze_139, buf437, buf438, 160, 19, grid=grid(160), stream=stream0)
        buf439 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_46, buf432, convolution_46, unsqueeze_942, buf437, squeeze_139, buf435, primals_93, buf439, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf432
        del convolution_46
        del primals_93
        del relu_46
        del squeeze_139
        del unsqueeze_942
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf440 = aten.convolution_backward(buf439, relu_45, primals_235, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_235
        buf441 = buf440[0]
        buf442 = buf440[1]
        del buf440
        buf443 = reinterpret_tensor(buf436, (160, 19), (19, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_45, buf441, buf443, 3040, 122, grid=grid(3040), stream=stream0)
        buf444 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf443, buf444, 160, 19, grid=grid(160), stream=stream0)
        buf445 = reinterpret_tensor(buf443, (160, 19), (1, 160), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_45, buf441, convolution_45, unsqueeze_954, buf445, 3040, 122, grid=grid(3040), stream=stream0)
        buf446 = empty((160, ), device='cuda', dtype=torch.float32)
        buf447 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf445, squeeze_136, buf446, buf447, 160, 19, grid=grid(160), stream=stream0)
        buf448 = buf439; del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_45, buf441, convolution_45, unsqueeze_954, buf446, squeeze_136, buf444, primals_91, buf448, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf441
        del convolution_45
        del primals_91
        del relu_45
        del squeeze_136
        del unsqueeze_954
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf449 = aten.convolution_backward(buf448, relu_44, primals_234, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_234
        buf450 = buf449[0]
        buf451 = buf449[1]
        del buf449
        buf452 = reinterpret_tensor(buf445, (160, 19), (19, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_44, buf450, buf452, 3040, 122, grid=grid(3040), stream=stream0)
        buf453 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf452, buf453, 160, 19, grid=grid(160), stream=stream0)
        buf454 = reinterpret_tensor(buf452, (160, 19), (1, 160), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_44, buf450, convolution_44, unsqueeze_966, buf454, 3040, 122, grid=grid(3040), stream=stream0)
        buf455 = empty((160, ), device='cuda', dtype=torch.float32)
        buf456 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf454, squeeze_133, buf455, buf456, 160, 19, grid=grid(160), stream=stream0)
        buf457 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_44, buf450, convolution_44, unsqueeze_966, buf455, squeeze_133, buf453, primals_89, buf457, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf450
        del convolution_44
        del primals_89
        del relu_44
        del squeeze_133
        del unsqueeze_966
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf458 = aten.convolution_backward(buf457, cat_4, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_233
        buf459 = buf458[0]
        buf460 = buf458[1]
        del buf458
        buf461 = reinterpret_tensor(buf421, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_76.run(le_50, buf324, buf368, buf395, buf404, buf461, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_50
        buf462 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf461, buf462, 192, 2312, grid=grid(192), stream=stream0)
        buf463 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf461, convolution_43, unsqueeze_978, buf463, 3648, 122, grid=grid(3648), stream=stream0)
        buf464 = empty((192, ), device='cuda', dtype=torch.float32)
        buf465 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf463, squeeze_130, buf464, buf465, 192, 19, grid=grid(192), stream=stream0)
        buf466 = reinterpret_tensor(buf416, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf461, convolution_43, unsqueeze_978, buf464, squeeze_130, buf462, primals_87, buf466, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_43
        del primals_87
        del squeeze_130
        del unsqueeze_978
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf467 = aten.convolution_backward(buf466, relu_42, primals_232, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_232
        buf468 = buf467[0]
        buf469 = buf467[1]
        del buf467
        buf470 = reinterpret_tensor(buf454, (160, 19), (19, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_42, buf468, buf470, 3040, 122, grid=grid(3040), stream=stream0)
        buf471 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf470, buf471, 160, 19, grid=grid(160), stream=stream0)
        buf472 = reinterpret_tensor(buf470, (160, 19), (1, 160), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_42, buf468, convolution_42, unsqueeze_990, buf472, 3040, 122, grid=grid(3040), stream=stream0)
        buf473 = empty((160, ), device='cuda', dtype=torch.float32)
        buf474 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf472, squeeze_127, buf473, buf474, 160, 19, grid=grid(160), stream=stream0)
        buf475 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_42, buf468, convolution_42, unsqueeze_990, buf473, squeeze_127, buf471, primals_85, buf475, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf468
        del convolution_42
        del primals_85
        del relu_42
        del squeeze_127
        del unsqueeze_990
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf476 = aten.convolution_backward(buf475, relu_41, primals_231, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_231
        buf477 = buf476[0]
        buf478 = buf476[1]
        del buf476
        buf479 = reinterpret_tensor(buf472, (160, 19), (19, 1), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_41, buf477, buf479, 3040, 122, grid=grid(3040), stream=stream0)
        buf480 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf479, buf480, 160, 19, grid=grid(160), stream=stream0)
        buf481 = reinterpret_tensor(buf479, (160, 19), (1, 160), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_73.run(relu_41, buf477, convolution_41, unsqueeze_1002, buf481, 3040, 122, grid=grid(3040), stream=stream0)
        buf482 = empty((160, ), device='cuda', dtype=torch.float32)
        buf483 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_74.run(buf481, squeeze_124, buf482, buf483, 160, 19, grid=grid(160), stream=stream0)
        del buf481
        buf484 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75.run(relu_41, buf477, convolution_41, unsqueeze_1002, buf482, squeeze_124, buf480, primals_83, buf484, 2312, 160, grid=grid(2312, 160), stream=stream0)
        del buf477
        del buf482
        del convolution_41
        del primals_83
        del relu_41
        del squeeze_124
        del unsqueeze_1002
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf485 = aten.convolution_backward(buf484, cat_4, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf484
        del primals_230
        buf486 = buf485[0]
        buf487 = buf485[1]
        del buf485
        buf488 = reinterpret_tensor(buf466, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_77.run(le_53, buf324, buf368, buf395, buf404, buf488, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del buf324
        del buf368
        del buf395
        del buf404
        del le_53
        buf489 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf488, buf489, 192, 2312, grid=grid(192), stream=stream0)
        buf490 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf488, convolution_40, unsqueeze_1014, buf490, 3648, 122, grid=grid(3648), stream=stream0)
        buf491 = empty((192, ), device='cuda', dtype=torch.float32)
        buf492 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf490, squeeze_121, buf491, buf492, 192, 19, grid=grid(192), stream=stream0)
        buf493 = reinterpret_tensor(buf461, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf488, convolution_40, unsqueeze_1014, buf491, squeeze_121, buf489, primals_81, buf493, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_40
        del primals_81
        del squeeze_121
        del unsqueeze_1014
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf494 = aten.convolution_backward(buf493, cat_4, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_4
        del primals_229
        buf495 = buf494[0]
        buf496 = buf494[1]
        del buf494
        buf497 = reinterpret_tensor(buf493, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_65.run(le_54, buf415, buf459, buf486, buf495, buf497, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_54
        buf498 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf497, buf498, 192, 2312, grid=grid(192), stream=stream0)
        buf499 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf497, convolution_39, unsqueeze_1026, buf499, 3648, 122, grid=grid(3648), stream=stream0)
        buf500 = empty((192, ), device='cuda', dtype=torch.float32)
        buf501 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf499, squeeze_118, buf500, buf501, 192, 19, grid=grid(192), stream=stream0)
        buf502 = reinterpret_tensor(buf488, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf497, convolution_39, unsqueeze_1026, buf500, squeeze_118, buf498, primals_79, buf502, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_39
        del primals_79
        del squeeze_118
        del unsqueeze_1026
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf503 = aten.convolution_backward(buf502, avg_pool2d_3, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_3
        del primals_228
        buf504 = buf503[0]
        buf505 = buf503[1]
        del buf503
        buf507 = reinterpret_tensor(buf502, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_70.run(le_55, buf415, buf459, buf486, buf495, buf507, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_55
        buf508 = buf500; del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf507, buf508, 192, 2312, grid=grid(192), stream=stream0)
        buf509 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf507, convolution_38, unsqueeze_1038, buf509, 3648, 122, grid=grid(3648), stream=stream0)
        buf510 = empty((192, ), device='cuda', dtype=torch.float32)
        buf511 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf509, squeeze_115, buf510, buf511, 192, 19, grid=grid(192), stream=stream0)
        buf512 = reinterpret_tensor(buf497, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf507, convolution_38, unsqueeze_1038, buf510, squeeze_115, buf508, primals_77, buf512, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_38
        del primals_77
        del squeeze_115
        del unsqueeze_1038
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf513 = aten.convolution_backward(buf512, relu_37, primals_227, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_227
        buf514 = buf513[0]
        buf516 = empty((128, 19), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_37, buf514, buf516, 2432, 122, grid=grid(2432), stream=stream0)
        buf517 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_79.run(buf516, buf517, 128, 19, grid=grid(128), stream=stream0)
        buf518 = reinterpret_tensor(buf516, (128, 19), (1, 128), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_37, buf514, convolution_37, unsqueeze_1050, buf518, 2432, 122, grid=grid(2432), stream=stream0)
        buf519 = empty((128, ), device='cuda', dtype=torch.float32)
        buf520 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf518, squeeze_112, buf519, buf520, 128, 19, grid=grid(128), stream=stream0)
        buf521 = empty_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82.run(relu_37, buf514, convolution_37, unsqueeze_1050, buf519, squeeze_112, buf517, primals_75, buf521, 2312, 128, grid=grid(2312, 128), stream=stream0)
        del buf514
        del convolution_37
        del primals_75
        del relu_37
        del squeeze_112
        del unsqueeze_1050
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf522 = aten.convolution_backward(buf521, relu_36, primals_226, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_226
        buf523 = buf522[0]
        buf525 = reinterpret_tensor(buf518, (128, 19), (19, 1), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_36, buf523, buf525, 2432, 122, grid=grid(2432), stream=stream0)
        buf526 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_79.run(buf525, buf526, 128, 19, grid=grid(128), stream=stream0)
        buf527 = reinterpret_tensor(buf525, (128, 19), (1, 128), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_36, buf523, convolution_36, unsqueeze_1062, buf527, 2432, 122, grid=grid(2432), stream=stream0)
        buf528 = empty((128, ), device='cuda', dtype=torch.float32)
        buf529 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf527, squeeze_109, buf528, buf529, 128, 19, grid=grid(128), stream=stream0)
        buf530 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82.run(relu_36, buf523, convolution_36, unsqueeze_1062, buf528, squeeze_109, buf526, primals_73, buf530, 2312, 128, grid=grid(2312, 128), stream=stream0)
        del buf523
        del convolution_36
        del primals_73
        del relu_36
        del squeeze_109
        del unsqueeze_1062
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf531 = aten.convolution_backward(buf530, relu_35, primals_225, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_225
        buf532 = buf531[0]
        buf534 = reinterpret_tensor(buf527, (128, 19), (19, 1), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_35, buf532, buf534, 2432, 122, grid=grid(2432), stream=stream0)
        buf535 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_79.run(buf534, buf535, 128, 19, grid=grid(128), stream=stream0)
        buf536 = reinterpret_tensor(buf534, (128, 19), (1, 128), 0); del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_35, buf532, convolution_35, unsqueeze_1074, buf536, 2432, 122, grid=grid(2432), stream=stream0)
        buf537 = empty((128, ), device='cuda', dtype=torch.float32)
        buf538 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf536, squeeze_106, buf537, buf538, 128, 19, grid=grid(128), stream=stream0)
        buf539 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82.run(relu_35, buf532, convolution_35, unsqueeze_1074, buf537, squeeze_106, buf535, primals_71, buf539, 2312, 128, grid=grid(2312, 128), stream=stream0)
        del buf532
        del convolution_35
        del primals_71
        del relu_35
        del squeeze_106
        del unsqueeze_1074
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf540 = aten.convolution_backward(buf539, relu_34, primals_224, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_224
        buf541 = buf540[0]
        buf543 = reinterpret_tensor(buf536, (128, 19), (19, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_34, buf541, buf543, 2432, 122, grid=grid(2432), stream=stream0)
        buf544 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_79.run(buf543, buf544, 128, 19, grid=grid(128), stream=stream0)
        buf545 = reinterpret_tensor(buf543, (128, 19), (1, 128), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_34, buf541, convolution_34, unsqueeze_1086, buf545, 2432, 122, grid=grid(2432), stream=stream0)
        buf546 = empty((128, ), device='cuda', dtype=torch.float32)
        buf547 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf545, squeeze_103, buf546, buf547, 128, 19, grid=grid(128), stream=stream0)
        buf548 = buf539; del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82.run(relu_34, buf541, convolution_34, unsqueeze_1086, buf546, squeeze_103, buf544, primals_69, buf548, 2312, 128, grid=grid(2312, 128), stream=stream0)
        del buf541
        del convolution_34
        del primals_69
        del relu_34
        del squeeze_103
        del unsqueeze_1086
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf549 = aten.convolution_backward(buf548, cat_3, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_223
        buf550 = buf549[0]
        buf552 = reinterpret_tensor(buf512, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_76.run(le_60, buf415, buf459, buf486, buf495, buf552, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del le_60
        buf553 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf552, buf553, 192, 2312, grid=grid(192), stream=stream0)
        buf554 = buf509; del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf552, convolution_33, unsqueeze_1098, buf554, 3648, 122, grid=grid(3648), stream=stream0)
        buf555 = empty((192, ), device='cuda', dtype=torch.float32)
        buf556 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf554, squeeze_100, buf555, buf556, 192, 19, grid=grid(192), stream=stream0)
        buf557 = reinterpret_tensor(buf507, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf552, convolution_33, unsqueeze_1098, buf555, squeeze_100, buf553, primals_67, buf557, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del convolution_33
        del primals_67
        del squeeze_100
        del unsqueeze_1098
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf558 = aten.convolution_backward(buf557, relu_32, primals_222, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_222
        buf559 = buf558[0]
        buf561 = reinterpret_tensor(buf545, (128, 19), (19, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_32, buf559, buf561, 2432, 122, grid=grid(2432), stream=stream0)
        buf562 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_79.run(buf561, buf562, 128, 19, grid=grid(128), stream=stream0)
        buf563 = reinterpret_tensor(buf561, (128, 19), (1, 128), 0); del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_32, buf559, convolution_32, unsqueeze_1110, buf563, 2432, 122, grid=grid(2432), stream=stream0)
        buf564 = empty((128, ), device='cuda', dtype=torch.float32)
        buf565 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf563, squeeze_97, buf564, buf565, 128, 19, grid=grid(128), stream=stream0)
        buf566 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82.run(relu_32, buf559, convolution_32, unsqueeze_1110, buf564, squeeze_97, buf562, primals_65, buf566, 2312, 128, grid=grid(2312, 128), stream=stream0)
        del buf559
        del convolution_32
        del primals_65
        del relu_32
        del squeeze_97
        del unsqueeze_1110
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf567 = aten.convolution_backward(buf566, relu_31, primals_221, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_221
        buf568 = buf567[0]
        buf570 = reinterpret_tensor(buf563, (128, 19), (19, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_31, buf568, buf570, 2432, 122, grid=grid(2432), stream=stream0)
        buf571 = buf564; del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_79.run(buf570, buf571, 128, 19, grid=grid(128), stream=stream0)
        buf572 = reinterpret_tensor(buf570, (128, 19), (1, 128), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_31, buf568, convolution_31, unsqueeze_1122, buf572, 2432, 122, grid=grid(2432), stream=stream0)
        buf573 = empty((128, ), device='cuda', dtype=torch.float32)
        buf574 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf572, squeeze_94, buf573, buf574, 128, 19, grid=grid(128), stream=stream0)
        del buf572
        buf575 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82.run(relu_31, buf568, convolution_31, unsqueeze_1122, buf573, squeeze_94, buf571, primals_63, buf575, 2312, 128, grid=grid(2312, 128), stream=stream0)
        del buf568
        del convolution_31
        del primals_63
        del relu_31
        del squeeze_94
        del unsqueeze_1122
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf576 = aten.convolution_backward(buf575, cat_3, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf575
        del primals_220
        buf577 = buf576[0]
        buf579 = reinterpret_tensor(buf557, (8, 192, 17, 17), (55488, 289, 17, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_77.run(le_63, buf415, buf459, buf486, buf495, buf579, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del buf415
        del buf459
        del buf486
        del le_63
        buf580 = buf555; del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf579, buf580, 192, 2312, grid=grid(192), stream=stream0)
        buf581 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_67.run(buf579, convolution_30, unsqueeze_1134, buf581, 3648, 122, grid=grid(3648), stream=stream0)
        buf582 = empty((192, ), device='cuda', dtype=torch.float32)
        buf583 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_68.run(buf581, squeeze_91, buf582, buf583, 192, 19, grid=grid(192), stream=stream0)
        del buf581
        buf584 = reinterpret_tensor(buf552, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf579, convolution_30, unsqueeze_1134, buf582, squeeze_91, buf580, primals_61, buf584, 2312, 192, grid=grid(2312, 192), stream=stream0)
        del buf579
        del convolution_30
        del primals_61
        del squeeze_91
        del unsqueeze_1134
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf585 = aten.convolution_backward(buf584, cat_3, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf584
        del cat_3
        del primals_219
        buf586 = buf585[0]
        buf506 = buf495; del buf495  # reuse
        buf588 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward]
        triton_poi_fused_add_avg_pool2d_backward_83.run(buf588, buf504, buf550, buf577, buf586, 1775616, grid=grid(1775616), stream=stream0)
        del buf504
        del buf550
        del buf577
        del buf586
        buf515 = buf513[1]
        del buf513
        buf524 = buf522[1]
        del buf522
        buf533 = buf531[1]
        del buf531
        buf542 = buf540[1]
        del buf540
        buf551 = buf549[1]
        del buf549
        buf560 = buf558[1]
        del buf558
        buf569 = buf567[1]
        del buf567
        buf578 = buf576[1]
        del buf576
        buf587 = buf585[1]
        del buf585
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf589 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf588, (8, 288, 17, 17), (221952, 289, 17, 1), 138720), cat_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_65)
        del getitem_65
        buf590 = buf589
        del buf589
        buf591 = empty((96, 19), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_84.run(le_64, buf588, buf591, 1824, 122, grid=grid(1824), stream=stream0)
        buf592 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_85.run(buf591, buf592, 96, 19, grid=grid(96), stream=stream0)
        buf593 = reinterpret_tensor(buf591, (96, 19), (1, 96), 0); del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_86.run(le_64, buf588, convolution_29, unsqueeze_1146, buf593, 1824, 122, grid=grid(1824), stream=stream0)
        buf594 = empty((96, ), device='cuda', dtype=torch.float32)
        buf595 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_87.run(buf593, squeeze_88, buf594, buf595, 96, 19, grid=grid(96), stream=stream0)
        del buf593
        buf596 = empty_strided((8, 96, 17, 17), (27744, 1, 1632, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_88.run(le_64, buf588, convolution_29, unsqueeze_1146, buf594, squeeze_88, buf592, primals_59, buf596, 2312, 96, grid=grid(2312, 96), stream=stream0)
        del convolution_29
        del le_64
        del primals_59
        del squeeze_88
        del unsqueeze_1146
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf597 = aten.convolution_backward(buf596, relu_28, primals_218, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf596
        del primals_218
        buf598 = buf597[0]
        buf599 = buf597[1]
        del buf597
        buf600 = empty((96, 77), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_28, buf598, buf600, 7392, 128, grid=grid(7392), stream=stream0)
        buf601 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_90.run(buf600, buf601, 96, 77, grid=grid(96), stream=stream0)
        buf602 = reinterpret_tensor(buf600, (96, 77), (1, 96), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_91.run(relu_28, buf598, convolution_28, unsqueeze_1158, buf602, 7392, 128, grid=grid(7392), stream=stream0)
        buf603 = empty((96, ), device='cuda', dtype=torch.float32)
        buf604 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_92.run(buf602, squeeze_85, buf603, buf604, 96, 77, grid=grid(96), stream=stream0)
        buf605 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93.run(relu_28, buf598, convolution_28, unsqueeze_1158, buf603, squeeze_85, buf601, primals_57, buf605, 9800, 96, grid=grid(9800, 96), stream=stream0)
        del buf598
        del convolution_28
        del primals_57
        del relu_28
        del squeeze_85
        del unsqueeze_1158
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf606 = aten.convolution_backward(buf605, relu_27, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_217
        buf607 = buf606[0]
        buf608 = buf606[1]
        del buf606
        buf609 = empty((64, 77), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_94.run(relu_27, buf607, buf609, 4928, 128, grid=grid(4928), stream=stream0)
        buf610 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_95.run(buf609, buf610, 64, 77, grid=grid(64), stream=stream0)
        buf611 = reinterpret_tensor(buf609, (64, 77), (1, 64), 0); del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_96.run(relu_27, buf607, convolution_27, unsqueeze_1170, buf611, 4928, 128, grid=grid(4928), stream=stream0)
        buf612 = empty((64, ), device='cuda', dtype=torch.float32)
        buf613 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(buf611, squeeze_82, buf612, buf613, 64, 77, grid=grid(64), stream=stream0)
        buf614 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_98.run(relu_27, buf607, convolution_27, unsqueeze_1170, buf612, squeeze_82, buf610, primals_55, buf614, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del buf607
        del convolution_27
        del primals_55
        del relu_27
        del squeeze_82
        del unsqueeze_1170
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf615 = aten.convolution_backward(buf614, cat_2, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_216
        buf616 = buf615[0]
        buf617 = buf615[1]
        del buf615
        buf618 = empty((384, 19), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_99.run(le_67, buf588, buf618, 7296, 122, grid=grid(7296), stream=stream0)
        buf619 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_100.run(buf618, buf619, 384, 19, grid=grid(384), stream=stream0)
        buf620 = reinterpret_tensor(buf618, (384, 19), (1, 384), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_101.run(le_67, buf588, convolution_26, unsqueeze_1182, buf620, 7296, 122, grid=grid(7296), stream=stream0)
        buf621 = empty((384, ), device='cuda', dtype=torch.float32)
        buf622 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_102.run(buf620, squeeze_79, buf621, buf622, 384, 19, grid=grid(384), stream=stream0)
        del buf620
        buf623 = empty_strided((8, 384, 17, 17), (110976, 1, 6528, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103.run(le_67, buf588, convolution_26, unsqueeze_1182, buf621, squeeze_79, buf619, primals_53, buf623, 2312, 384, grid=grid(2312, 384), stream=stream0)
        del buf588
        del buf621
        del convolution_26
        del le_67
        del primals_53
        del squeeze_79
        del unsqueeze_1182
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf624 = aten.convolution_backward(buf623, cat_2, primals_215, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf623
        del cat_2
        del primals_215
        buf625 = buf624[0]
        buf626 = buf624[1]
        del buf624
        buf627 = reinterpret_tensor(buf611, (64, 77), (77, 1), 0); del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_104.run(le_68, buf590, buf616, buf625, buf627, 4928, 128, grid=grid(4928), stream=stream0)
        buf628 = buf612; del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_95.run(buf627, buf628, 64, 77, grid=grid(64), stream=stream0)
        buf629 = reinterpret_tensor(buf627, (64, 77), (1, 64), 0); del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_105.run(le_68, buf590, buf616, buf625, convolution_25, unsqueeze_1194, buf629, 4928, 128, grid=grid(4928), stream=stream0)
        buf630 = empty((64, ), device='cuda', dtype=torch.float32)
        buf632 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(buf629, squeeze_76, buf630, buf632, 64, 77, grid=grid(64), stream=stream0)
        buf631 = buf614; del buf614  # reuse
        buf633 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_106.run(buf633, le_68, buf590, buf616, buf625, convolution_25, unsqueeze_1194, buf630, squeeze_76, buf628, primals_51, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del convolution_25
        del le_68
        del primals_51
        del squeeze_76
        del unsqueeze_1194
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf634 = aten.convolution_backward(buf633, avg_pool2d_2, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_2
        del primals_214
        buf635 = buf634[0]
        buf636 = buf634[1]
        del buf634
        buf637 = empty((8, 288, 35, 35), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_107.run(buf635, buf637, 2822400, grid=grid(2822400), stream=stream0)
        del buf635
        buf638 = reinterpret_tensor(buf602, (96, 77), (77, 1), 0); del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_108.run(le_69, buf590, buf616, buf625, buf638, 7392, 128, grid=grid(7392), stream=stream0)
        buf639 = buf603; del buf603  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_90.run(buf638, buf639, 96, 77, grid=grid(96), stream=stream0)
        buf640 = reinterpret_tensor(buf638, (96, 77), (1, 96), 0); del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_109.run(le_69, buf590, buf616, buf625, convolution_24, unsqueeze_1206, buf640, 7392, 128, grid=grid(7392), stream=stream0)
        buf641 = empty((96, ), device='cuda', dtype=torch.float32)
        buf643 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_92.run(buf640, squeeze_73, buf641, buf643, 96, 77, grid=grid(96), stream=stream0)
        buf642 = buf605; del buf605  # reuse
        buf644 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_110.run(buf644, le_69, buf590, buf616, buf625, convolution_24, unsqueeze_1206, buf641, squeeze_73, buf639, primals_49, 9800, 96, grid=grid(9800, 96), stream=stream0)
        del convolution_24
        del le_69
        del primals_49
        del squeeze_73
        del unsqueeze_1206
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf645 = aten.convolution_backward(buf644, relu_23, primals_213, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_213
        buf646 = buf645[0]
        buf647 = buf645[1]
        del buf645
        buf648 = reinterpret_tensor(buf640, (96, 77), (77, 1), 0); del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_23, buf646, buf648, 7392, 128, grid=grid(7392), stream=stream0)
        buf649 = buf641; del buf641  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_90.run(buf648, buf649, 96, 77, grid=grid(96), stream=stream0)
        buf650 = reinterpret_tensor(buf648, (96, 77), (1, 96), 0); del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_91.run(relu_23, buf646, convolution_23, unsqueeze_1218, buf650, 7392, 128, grid=grid(7392), stream=stream0)
        buf651 = empty((96, ), device='cuda', dtype=torch.float32)
        buf652 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_92.run(buf650, squeeze_70, buf651, buf652, 96, 77, grid=grid(96), stream=stream0)
        buf653 = buf644; del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93.run(relu_23, buf646, convolution_23, unsqueeze_1218, buf651, squeeze_70, buf649, primals_47, buf653, 9800, 96, grid=grid(9800, 96), stream=stream0)
        del convolution_23
        del primals_47
        del relu_23
        del squeeze_70
        del unsqueeze_1218
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf654 = aten.convolution_backward(buf653, relu_22, primals_212, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_212
        buf655 = buf654[0]
        buf656 = buf654[1]
        del buf654
        buf657 = reinterpret_tensor(buf629, (64, 77), (77, 1), 0); del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_94.run(relu_22, buf655, buf657, 4928, 128, grid=grid(4928), stream=stream0)
        buf658 = buf630; del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_95.run(buf657, buf658, 64, 77, grid=grid(64), stream=stream0)
        buf659 = reinterpret_tensor(buf657, (64, 77), (1, 64), 0); del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_96.run(relu_22, buf655, convolution_22, unsqueeze_1230, buf659, 4928, 128, grid=grid(4928), stream=stream0)
        buf660 = empty((64, ), device='cuda', dtype=torch.float32)
        buf661 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(buf659, squeeze_67, buf660, buf661, 64, 77, grid=grid(64), stream=stream0)
        buf662 = buf633; del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_98.run(relu_22, buf655, convolution_22, unsqueeze_1230, buf660, squeeze_67, buf658, primals_45, buf662, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del convolution_22
        del primals_45
        del relu_22
        del squeeze_67
        del unsqueeze_1230
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf663 = aten.convolution_backward(buf662, cat_1, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_211
        buf664 = buf663[0]
        buf665 = buf663[1]
        del buf663
        buf666 = reinterpret_tensor(buf659, (64, 77), (77, 1), 0); del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_111.run(le_72, buf590, buf616, buf625, buf666, 4928, 128, grid=grid(4928), stream=stream0)
        buf667 = buf660; del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_95.run(buf666, buf667, 64, 77, grid=grid(64), stream=stream0)
        buf668 = reinterpret_tensor(buf666, (64, 77), (1, 64), 0); del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_112.run(le_72, buf590, buf616, buf625, convolution_21, unsqueeze_1242, buf668, 4928, 128, grid=grid(4928), stream=stream0)
        buf669 = empty((64, ), device='cuda', dtype=torch.float32)
        buf671 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(buf668, squeeze_64, buf669, buf671, 64, 77, grid=grid(64), stream=stream0)
        buf670 = buf662; del buf662  # reuse
        buf672 = buf670; del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_113.run(buf672, le_72, buf590, buf616, buf625, convolution_21, unsqueeze_1242, buf669, squeeze_64, buf667, primals_43, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del convolution_21
        del le_72
        del primals_43
        del squeeze_64
        del unsqueeze_1242
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf673 = aten.convolution_backward(buf672, relu_20, primals_210, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_210
        buf674 = buf673[0]
        buf675 = buf673[1]
        del buf673
        buf676 = empty((48, 77), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_114.run(relu_20, buf674, buf676, 3696, 128, grid=grid(3696), stream=stream0)
        buf677 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_115.run(buf676, buf677, 48, 77, grid=grid(48), stream=stream0)
        buf678 = reinterpret_tensor(buf676, (48, 77), (1, 48), 0); del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_116.run(relu_20, buf674, convolution_20, unsqueeze_1254, buf678, 3696, 128, grid=grid(3696), stream=stream0)
        buf679 = empty((48, ), device='cuda', dtype=torch.float32)
        buf680 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_117.run(buf678, squeeze_61, buf679, buf680, 48, 77, grid=grid(48), stream=stream0)
        buf681 = empty_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_118.run(relu_20, buf674, convolution_20, unsqueeze_1254, buf679, squeeze_61, buf677, primals_41, buf681, 9800, 48, grid=grid(9800, 48), stream=stream0)
        del buf674
        del convolution_20
        del primals_41
        del relu_20
        del squeeze_61
        del unsqueeze_1254
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf682 = aten.convolution_backward(buf681, cat_1, primals_209, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_209
        buf683 = buf682[0]
        buf684 = buf682[1]
        del buf682
        buf685 = reinterpret_tensor(buf668, (64, 77), (77, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_119.run(le_74, buf590, buf616, buf625, buf685, 4928, 128, grid=grid(4928), stream=stream0)
        buf686 = buf669; del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_95.run(buf685, buf686, 64, 77, grid=grid(64), stream=stream0)
        buf687 = reinterpret_tensor(buf685, (64, 77), (1, 64), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(le_74, buf590, buf616, buf625, convolution_19, unsqueeze_1266, buf687, 4928, 128, grid=grid(4928), stream=stream0)
        buf688 = empty((64, ), device='cuda', dtype=torch.float32)
        buf690 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(buf687, squeeze_58, buf688, buf690, 64, 77, grid=grid(64), stream=stream0)
        buf689 = buf672; del buf672  # reuse
        buf691 = buf689; del buf689  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_121.run(buf691, le_74, buf590, buf616, buf625, convolution_19, unsqueeze_1266, buf688, squeeze_58, buf686, primals_39, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del buf590
        del buf616
        del buf625
        del convolution_19
        del le_74
        del primals_39
        del squeeze_58
        del unsqueeze_1266
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf692 = aten.convolution_backward(buf691, cat_1, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_208
        buf693 = buf692[0]
        buf694 = buf692[1]
        del buf692
        buf695 = reinterpret_tensor(buf691, (8, 64, 35, 35), (78400, 1225, 35, 1), 0); del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_122.run(le_75, buf637, buf664, buf683, buf693, buf695, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del le_75
        buf696 = reinterpret_tensor(buf573, (64, 2), (1, 64), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_123.run(buf695, buf696, 128, 4900, grid=grid(128), stream=stream0)
        buf697 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_124.run(buf696, buf697, 64, 2, grid=grid(64), stream=stream0)
        buf698 = reinterpret_tensor(buf687, (64, 77), (77, 1), 0); del buf687  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_125.run(buf695, convolution_18, unsqueeze_1278, buf698, 4928, 128, grid=grid(4928), stream=stream0)
        buf699 = empty((64, ), device='cuda', dtype=torch.float32)
        buf700 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf698, squeeze_55, buf699, buf700, 64, 77, grid=grid(64), stream=stream0)
        buf701 = reinterpret_tensor(buf655, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_127.run(buf695, convolution_18, unsqueeze_1278, buf699, squeeze_55, buf697, primals_37, buf701, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del buf695
        del convolution_18
        del primals_37
        del squeeze_55
        del unsqueeze_1278
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf702 = aten.convolution_backward(buf701, avg_pool2d_1, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d_1
        del primals_207
        buf703 = buf702[0]
        buf704 = buf702[1]
        del buf702
        buf705 = empty((8, 256, 35, 35), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_128.run(buf703, buf705, 2508800, grid=grid(2508800), stream=stream0)
        del buf703
        buf706 = reinterpret_tensor(buf653, (8, 96, 35, 35), (117600, 1225, 35, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_129.run(le_76, buf637, buf664, buf683, buf693, buf706, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del le_76
        buf707 = reinterpret_tensor(buf582, (96, 2), (1, 96), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_130.run(buf706, buf707, 192, 4900, grid=grid(192), stream=stream0)
        buf708 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_131.run(buf707, buf708, 96, 2, grid=grid(96), stream=stream0)
        buf709 = reinterpret_tensor(buf650, (96, 77), (77, 1), 0); del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_132.run(buf706, convolution_17, unsqueeze_1290, buf709, 7392, 128, grid=grid(7392), stream=stream0)
        buf710 = empty((96, ), device='cuda', dtype=torch.float32)
        buf711 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_133.run(buf709, squeeze_52, buf710, buf711, 96, 77, grid=grid(96), stream=stream0)
        buf712 = reinterpret_tensor(buf646, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf646  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_134.run(buf706, convolution_17, unsqueeze_1290, buf710, squeeze_52, buf708, primals_35, buf712, 9800, 96, grid=grid(9800, 96), stream=stream0)
        del buf706
        del convolution_17
        del primals_35
        del squeeze_52
        del unsqueeze_1290
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf713 = aten.convolution_backward(buf712, relu_16, primals_206, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_206
        buf714 = buf713[0]
        buf715 = buf713[1]
        del buf713
        buf716 = buf709; del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_16, buf714, buf716, 7392, 128, grid=grid(7392), stream=stream0)
        buf717 = buf710; del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_90.run(buf716, buf717, 96, 77, grid=grid(96), stream=stream0)
        buf718 = reinterpret_tensor(buf716, (96, 77), (1, 96), 0); del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_91.run(relu_16, buf714, convolution_16, unsqueeze_1302, buf718, 7392, 128, grid=grid(7392), stream=stream0)
        buf719 = empty((96, ), device='cuda', dtype=torch.float32)
        buf720 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_92.run(buf718, squeeze_49, buf719, buf720, 96, 77, grid=grid(96), stream=stream0)
        buf721 = buf712; del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93.run(relu_16, buf714, convolution_16, unsqueeze_1302, buf719, squeeze_49, buf717, primals_33, buf721, 9800, 96, grid=grid(9800, 96), stream=stream0)
        del convolution_16
        del primals_33
        del relu_16
        del squeeze_49
        del unsqueeze_1302
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf722 = aten.convolution_backward(buf721, relu_15, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_205
        buf723 = buf722[0]
        buf724 = buf722[1]
        del buf722
        buf725 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_94.run(relu_15, buf723, buf725, 4928, 128, grid=grid(4928), stream=stream0)
        buf726 = buf699; del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_95.run(buf725, buf726, 64, 77, grid=grid(64), stream=stream0)
        buf727 = reinterpret_tensor(buf725, (64, 77), (1, 64), 0); del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_96.run(relu_15, buf723, convolution_15, unsqueeze_1314, buf727, 4928, 128, grid=grid(4928), stream=stream0)
        buf728 = empty((64, ), device='cuda', dtype=torch.float32)
        buf729 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(buf727, squeeze_46, buf728, buf729, 64, 77, grid=grid(64), stream=stream0)
        buf730 = buf701; del buf701  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_98.run(relu_15, buf723, convolution_15, unsqueeze_1314, buf728, squeeze_46, buf726, primals_31, buf730, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del convolution_15
        del primals_31
        del relu_15
        del squeeze_46
        del unsqueeze_1314
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf731 = aten.convolution_backward(buf730, cat, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_204
        buf732 = buf731[0]
        buf733 = buf731[1]
        del buf731
        buf734 = reinterpret_tensor(buf730, (8, 64, 35, 35), (78400, 1225, 35, 1), 0); del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_135.run(le_79, buf637, buf664, buf683, buf693, buf734, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del le_79
        buf735 = buf696; del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_123.run(buf734, buf735, 128, 4900, grid=grid(128), stream=stream0)
        buf736 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_124.run(buf735, buf736, 64, 2, grid=grid(64), stream=stream0)
        buf737 = reinterpret_tensor(buf727, (64, 77), (77, 1), 0); del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_125.run(buf734, convolution_14, unsqueeze_1326, buf737, 4928, 128, grid=grid(4928), stream=stream0)
        buf738 = empty((64, ), device='cuda', dtype=torch.float32)
        buf739 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf737, squeeze_43, buf738, buf739, 64, 77, grid=grid(64), stream=stream0)
        buf740 = reinterpret_tensor(buf723, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_127.run(buf734, convolution_14, unsqueeze_1326, buf738, squeeze_43, buf736, primals_29, buf740, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del convolution_14
        del primals_29
        del squeeze_43
        del unsqueeze_1326
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf741 = aten.convolution_backward(buf740, relu_13, primals_203, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_203
        buf742 = buf741[0]
        buf743 = buf741[1]
        del buf741
        buf744 = reinterpret_tensor(buf678, (48, 77), (77, 1), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_114.run(relu_13, buf742, buf744, 3696, 128, grid=grid(3696), stream=stream0)
        buf745 = buf679; del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_115.run(buf744, buf745, 48, 77, grid=grid(48), stream=stream0)
        buf746 = reinterpret_tensor(buf744, (48, 77), (1, 48), 0); del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_116.run(relu_13, buf742, convolution_13, unsqueeze_1338, buf746, 3696, 128, grid=grid(3696), stream=stream0)
        buf747 = empty((48, ), device='cuda', dtype=torch.float32)
        buf748 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_117.run(buf746, squeeze_40, buf747, buf748, 48, 77, grid=grid(48), stream=stream0)
        buf749 = buf681; del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_118.run(relu_13, buf742, convolution_13, unsqueeze_1338, buf747, squeeze_40, buf745, primals_27, buf749, 9800, 48, grid=grid(9800, 48), stream=stream0)
        del buf742
        del convolution_13
        del primals_27
        del relu_13
        del squeeze_40
        del unsqueeze_1338
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf750 = aten.convolution_backward(buf749, cat, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_202
        buf751 = buf750[0]
        buf752 = buf750[1]
        del buf750
        buf753 = reinterpret_tensor(buf740, (8, 64, 35, 35), (78400, 1225, 35, 1), 0); del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_136.run(le_81, buf637, buf664, buf683, buf693, buf753, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del buf637
        del buf664
        del buf683
        del buf693
        del le_81
        buf754 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_123.run(buf753, buf754, 128, 4900, grid=grid(128), stream=stream0)
        buf755 = buf738; del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_124.run(buf754, buf755, 64, 2, grid=grid(64), stream=stream0)
        buf756 = buf737; del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_125.run(buf753, convolution_12, unsqueeze_1350, buf756, 4928, 128, grid=grid(4928), stream=stream0)
        buf757 = empty((64, ), device='cuda', dtype=torch.float32)
        buf758 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf756, squeeze_37, buf757, buf758, 64, 77, grid=grid(64), stream=stream0)
        buf759 = reinterpret_tensor(buf734, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_127.run(buf753, convolution_12, unsqueeze_1350, buf757, squeeze_37, buf755, primals_25, buf759, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del buf753
        del convolution_12
        del primals_25
        del squeeze_37
        del unsqueeze_1350
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf760 = aten.convolution_backward(buf759, cat, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat
        del primals_201
        buf761 = buf760[0]
        buf762 = buf760[1]
        del buf760
        buf763 = empty((8, 32, 35, 35), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_137.run(le_82, buf705, buf732, buf751, buf761, buf763, 256, 1225, grid=grid(256, 1225), stream=stream0)
        del le_82
        buf764 = reinterpret_tensor(buf757, (32, 2), (1, 32), 0); del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_138.run(buf763, buf764, 64, 4900, grid=grid(64), stream=stream0)
        buf765 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_139.run(buf764, buf765, 32, 2, grid=grid(32), stream=stream0)
        buf766 = empty((32, 77), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_140.run(buf763, convolution_11, unsqueeze_1362, buf766, 2464, 128, grid=grid(2464), stream=stream0)
        buf767 = empty((32, ), device='cuda', dtype=torch.float32)
        buf768 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_141.run(buf766, squeeze_34, buf767, buf768, 32, 77, grid=grid(32), stream=stream0)
        del buf766
        buf769 = empty_strided((8, 32, 35, 35), (39200, 1, 1120, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_142.run(buf763, convolution_11, unsqueeze_1362, buf767, squeeze_34, buf765, primals_23, buf769, 9800, 32, grid=grid(9800, 32), stream=stream0)
        del buf763
        del convolution_11
        del primals_23
        del squeeze_34
        del unsqueeze_1362
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf770 = aten.convolution_backward(buf769, avg_pool2d, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del avg_pool2d
        del buf769
        del primals_200
        buf771 = buf770[0]
        buf772 = buf770[1]
        del buf770
        buf774 = reinterpret_tensor(buf721, (8, 96, 35, 35), (117600, 1225, 35, 1), 0); del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_143.run(le_83, buf705, buf732, buf751, buf761, buf774, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del le_83
        buf775 = buf707; del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_130.run(buf774, buf775, 192, 4900, grid=grid(192), stream=stream0)
        buf776 = buf719; del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_131.run(buf775, buf776, 96, 2, grid=grid(96), stream=stream0)
        buf777 = reinterpret_tensor(buf718, (96, 77), (77, 1), 0); del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_132.run(buf774, convolution_10, unsqueeze_1374, buf777, 7392, 128, grid=grid(7392), stream=stream0)
        buf778 = empty((96, ), device='cuda', dtype=torch.float32)
        buf779 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_133.run(buf777, squeeze_31, buf778, buf779, 96, 77, grid=grid(96), stream=stream0)
        buf780 = reinterpret_tensor(buf714, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_134.run(buf774, convolution_10, unsqueeze_1374, buf778, squeeze_31, buf776, primals_21, buf780, 9800, 96, grid=grid(9800, 96), stream=stream0)
        del buf774
        del convolution_10
        del primals_21
        del squeeze_31
        del unsqueeze_1374
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf781 = aten.convolution_backward(buf780, relu_9, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_199
        buf782 = buf781[0]
        buf784 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_89.run(relu_9, buf782, buf784, 7392, 128, grid=grid(7392), stream=stream0)
        buf785 = buf778; del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_90.run(buf784, buf785, 96, 77, grid=grid(96), stream=stream0)
        buf786 = reinterpret_tensor(buf784, (96, 77), (1, 96), 0); del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_91.run(relu_9, buf782, convolution_9, unsqueeze_1386, buf786, 7392, 128, grid=grid(7392), stream=stream0)
        buf787 = empty((96, ), device='cuda', dtype=torch.float32)
        buf788 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_92.run(buf786, squeeze_28, buf787, buf788, 96, 77, grid=grid(96), stream=stream0)
        del buf786
        buf789 = buf780; del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93.run(relu_9, buf782, convolution_9, unsqueeze_1386, buf787, squeeze_28, buf785, primals_19, buf789, 9800, 96, grid=grid(9800, 96), stream=stream0)
        del buf782
        del buf787
        del convolution_9
        del primals_19
        del relu_9
        del squeeze_28
        del unsqueeze_1386
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf790 = aten.convolution_backward(buf789, relu_8, primals_198, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf789
        del primals_198
        buf791 = buf790[0]
        buf793 = buf756; del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_94.run(relu_8, buf791, buf793, 4928, 128, grid=grid(4928), stream=stream0)
        buf794 = reinterpret_tensor(buf764, (64, ), (1, ), 0); del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_95.run(buf793, buf794, 64, 77, grid=grid(64), stream=stream0)
        buf795 = reinterpret_tensor(buf793, (64, 77), (1, 64), 0); del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_96.run(relu_8, buf791, convolution_8, unsqueeze_1398, buf795, 4928, 128, grid=grid(4928), stream=stream0)
        buf796 = empty((64, ), device='cuda', dtype=torch.float32)
        buf797 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_97.run(buf795, squeeze_25, buf796, buf797, 64, 77, grid=grid(64), stream=stream0)
        buf798 = buf759; del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_98.run(relu_8, buf791, convolution_8, unsqueeze_1398, buf796, squeeze_25, buf794, primals_17, buf798, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del convolution_8
        del primals_17
        del relu_8
        del squeeze_25
        del unsqueeze_1398
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf799 = aten.convolution_backward(buf798, getitem_12, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_197
        buf800 = buf799[0]
        buf802 = reinterpret_tensor(buf798, (8, 64, 35, 35), (78400, 1225, 35, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_144.run(le_86, buf705, buf732, buf751, buf761, buf802, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del le_86
        buf803 = buf754; del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_123.run(buf802, buf803, 128, 4900, grid=grid(128), stream=stream0)
        buf804 = buf796; del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_124.run(buf803, buf804, 64, 2, grid=grid(64), stream=stream0)
        buf805 = reinterpret_tensor(buf795, (64, 77), (77, 1), 0); del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_125.run(buf802, convolution_7, unsqueeze_1410, buf805, 4928, 128, grid=grid(4928), stream=stream0)
        buf806 = empty((64, ), device='cuda', dtype=torch.float32)
        buf807 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf805, squeeze_22, buf806, buf807, 64, 77, grid=grid(64), stream=stream0)
        buf808 = reinterpret_tensor(buf791, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_127.run(buf802, convolution_7, unsqueeze_1410, buf806, squeeze_22, buf804, primals_15, buf808, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del convolution_7
        del primals_15
        del squeeze_22
        del unsqueeze_1410
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf809 = aten.convolution_backward(buf808, relu_6, primals_196, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_196
        buf810 = buf809[0]
        buf812 = reinterpret_tensor(buf746, (48, 77), (77, 1), 0); del buf746  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_114.run(relu_6, buf810, buf812, 3696, 128, grid=grid(3696), stream=stream0)
        buf813 = buf747; del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_115.run(buf812, buf813, 48, 77, grid=grid(48), stream=stream0)
        buf814 = reinterpret_tensor(buf812, (48, 77), (1, 48), 0); del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_116.run(relu_6, buf810, convolution_6, unsqueeze_1422, buf814, 3696, 128, grid=grid(3696), stream=stream0)
        buf815 = empty((48, ), device='cuda', dtype=torch.float32)
        buf816 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_117.run(buf814, squeeze_19, buf815, buf816, 48, 77, grid=grid(48), stream=stream0)
        del buf814
        buf817 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_118.run(relu_6, buf810, convolution_6, unsqueeze_1422, buf815, squeeze_19, buf813, primals_13, buf817, 9800, 48, grid=grid(9800, 48), stream=stream0)
        del buf810
        del buf815
        del convolution_6
        del primals_13
        del relu_6
        del squeeze_19
        del unsqueeze_1422
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf818 = aten.convolution_backward(buf817, getitem_12, primals_195, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf817
        del primals_195
        buf819 = buf818[0]
        buf821 = reinterpret_tensor(buf808, (8, 64, 35, 35), (78400, 1225, 35, 1), 0); del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_145.run(le_88, buf705, buf732, buf751, buf761, buf821, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del buf705
        del buf732
        del buf751
        del buf761
        del le_88
        buf822 = buf803; del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_123.run(buf821, buf822, 128, 4900, grid=grid(128), stream=stream0)
        buf823 = buf806; del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_124.run(buf822, buf823, 64, 2, grid=grid(64), stream=stream0)
        del buf822
        buf824 = buf805; del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_125.run(buf821, convolution_5, unsqueeze_1434, buf824, 4928, 128, grid=grid(4928), stream=stream0)
        buf825 = empty((64, ), device='cuda', dtype=torch.float32)
        buf826 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf824, squeeze_16, buf825, buf826, 64, 77, grid=grid(64), stream=stream0)
        del buf824
        buf827 = reinterpret_tensor(buf802, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_127.run(buf821, convolution_5, unsqueeze_1434, buf825, squeeze_16, buf823, primals_11, buf827, 9800, 64, grid=grid(9800, 64), stream=stream0)
        del buf821
        del convolution_5
        del primals_11
        del squeeze_16
        del unsqueeze_1434
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf828 = aten.convolution_backward(buf827, getitem_12, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf827
        del getitem_12
        del primals_194
        buf829 = buf828[0]
        buf773 = empty((8, 192, 35, 35), device='cuda', dtype=torch.float32)
        buf831 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward]
        triton_poi_fused_add_avg_pool2d_backward_146.run(buf831, buf771, buf800, buf819, buf829, 1881600, grid=grid(1881600), stream=stream0)
        del buf771
        del buf800
        del buf819
        del buf829
        buf783 = buf781[1]
        del buf781
        buf792 = buf790[1]
        del buf790
        buf801 = buf799[1]
        del buf799
        buf811 = buf809[1]
        del buf809
        buf820 = buf818[1]
        del buf818
        buf830 = buf828[1]
        del buf828
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf832 = aten.max_pool2d_with_indices_backward(buf831, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_13)
        del buf831
        del getitem_13
        buf833 = buf832
        del buf832
        buf834 = empty_strided((192, 316), (1, 192), device='cuda', dtype=torch.float32)
        buf836 = empty_strided((192, 316), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_147.run(relu_4, buf833, convolution_4, unsqueeze_1446, buf834, buf836, 60672, 128, grid=grid(60672), stream=stream0)
        buf835 = reinterpret_tensor(buf775, (192, ), (1, ), 0); del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_148.run(buf834, buf835, 192, 316, grid=grid(192), stream=stream0)
        del buf834
        buf837 = empty((192, ), device='cuda', dtype=torch.float32)
        buf838 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_149.run(buf836, squeeze_13, buf837, buf838, 192, 316, grid=grid(192), stream=stream0)
        del buf836
        buf839 = buf833; del buf833  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_150.run(buf839, relu_4, convolution_4, unsqueeze_1446, buf837, squeeze_13, buf835, primals_9, 7742976, grid=grid(7742976), stream=stream0)
        del buf837
        del convolution_4
        del primals_9
        del relu_4
        del squeeze_13
        del unsqueeze_1446
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf840 = aten.convolution_backward(buf839, relu_3, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf839
        del primals_193
        buf841 = buf840[0]
        buf842 = buf840[1]
        del buf840
        buf843 = empty((80, 334), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_151.run(relu_3, buf841, buf843, 26720, 128, grid=grid(26720), stream=stream0)
        buf844 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_152.run(buf843, buf844, 80, 334, grid=grid(80), stream=stream0)
        buf845 = reinterpret_tensor(buf843, (80, 334), (1, 80), 0); del buf843  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_153.run(relu_3, buf841, convolution_3, unsqueeze_1458, buf845, 26720, 128, grid=grid(26720), stream=stream0)
        buf846 = empty((80, ), device='cuda', dtype=torch.float32)
        buf847 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_154.run(buf845, squeeze_10, buf846, buf847, 80, 334, grid=grid(80), stream=stream0)
        del buf845
        buf848 = empty_strided((8, 80, 73, 73), (426320, 1, 5840, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_155.run(relu_3, buf841, convolution_3, unsqueeze_1458, buf846, squeeze_10, buf844, primals_7, buf848, 42632, 80, grid=grid(42632, 80), stream=stream0)
        del buf841
        del buf846
        del convolution_3
        del primals_7
        del relu_3
        del squeeze_10
        del unsqueeze_1458
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf849 = aten.convolution_backward(buf848, getitem_6, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf848
        del getitem_6
        del primals_192
        buf850 = buf849[0]
        buf851 = buf849[1]
        del buf849
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf852 = aten.max_pool2d_with_indices_backward(buf850, relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_7)
        del buf850
        del getitem_7
        buf853 = buf852
        del buf852
        buf854 = empty_strided((64, 1029), (1, 64), device='cuda', dtype=torch.float32)
        buf856 = empty_strided((64, 1029), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_156.run(relu_2, buf853, convolution_2, unsqueeze_1470, buf854, buf856, 65856, 168, grid=grid(65856), stream=stream0)
        buf855 = buf825; del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_157.run(buf854, buf855, 64, 1029, grid=grid(64), stream=stream0)
        del buf854
        buf857 = empty((64, ), device='cuda', dtype=torch.float32)
        buf858 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_158.run(buf856, squeeze_7, buf857, buf858, 64, 1029, grid=grid(64), stream=stream0)
        del buf856
        buf859 = buf853; del buf853  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_159.run(buf859, relu_2, convolution_2, unsqueeze_1470, buf857, squeeze_7, buf855, primals_5, 11063808, grid=grid(11063808), stream=stream0)
        del buf857
        del convolution_2
        del primals_5
        del relu_2
        del squeeze_7
        del unsqueeze_1470
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf860 = aten.convolution_backward(buf859, relu_1, primals_191, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf859
        del primals_191
        buf861 = buf860[0]
        buf862 = buf860[1]
        del buf860
        buf863 = empty((32, 1351), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_160.run(relu_1, buf861, buf863, 43232, 128, grid=grid(43232), stream=stream0)
        buf864 = buf767; del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_161.run(buf863, buf864, 32, 1351, grid=grid(32), stream=stream0)
        buf865 = reinterpret_tensor(buf863, (32, 1351), (1, 32), 0); del buf863  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_162.run(relu_1, buf861, convolution_1, unsqueeze_1482, buf865, 43232, 128, grid=grid(43232), stream=stream0)
        buf866 = empty((32, ), device='cuda', dtype=torch.float32)
        buf867 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_163.run(buf865, squeeze_4, buf866, buf867, 32, 1351, grid=grid(32), stream=stream0)
        del buf865
        buf868 = empty_strided((8, 32, 147, 147), (691488, 1, 4704, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_164.run(relu_1, buf861, convolution_1, unsqueeze_1482, buf866, squeeze_4, buf864, primals_3, buf868, 172872, 32, grid=grid(172872, 32), stream=stream0)
        del buf861
        del convolution_1
        del primals_3
        del relu_1
        del squeeze_4
        del unsqueeze_1482
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf869 = aten.convolution_backward(buf868, relu, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf868
        del primals_190
        buf870 = buf869[0]
        buf871 = buf869[1]
        del buf869
        buf872 = empty((32, 1388), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_165.run(relu, buf870, buf872, 44416, 128, grid=grid(44416), stream=stream0)
        buf873 = buf866; del buf866  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_166.run(buf872, buf873, 32, 1388, grid=grid(32), stream=stream0)
        buf874 = reinterpret_tensor(buf872, (32, 1388), (1, 32), 0); del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_167.run(relu, buf870, convolution, unsqueeze_1494, buf874, 44416, 128, grid=grid(44416), stream=stream0)
        buf875 = empty((32, ), device='cuda', dtype=torch.float32)
        buf876 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_168.run(buf874, squeeze_1, buf875, buf876, 32, 1388, grid=grid(32), stream=stream0)
        del buf874
        buf877 = empty_strided((8, 32, 149, 149), (710432, 1, 4768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_169.run(relu, buf870, convolution, unsqueeze_1494, buf875, squeeze_1, buf873, primals_1, buf877, 177608, 32, grid=grid(177608, 32), stream=stream0)
        del buf870
        del buf875
        del convolution
        del primals_1
        del relu
        del squeeze_1
        del unsqueeze_1494
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf878 = aten.convolution_backward(buf877, primals_567, primals_189, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf877
        del primals_189
        del primals_567
        buf879 = buf878[1]
        return (buf876, buf873, buf867, buf864, buf858, buf855, buf847, buf844, buf838, buf835, buf826, buf823, buf816, buf813, buf807, buf804, buf797, buf794, buf788, buf785, buf779, buf776, buf768, buf765, buf758, buf755, buf748, buf745, buf739, buf736, buf729, buf726, buf720, buf717, buf711, buf708, buf700, buf697, buf690, buf686, buf680, buf677, buf671, buf667, buf661, buf658, buf652, buf649, buf643, buf639, buf632, buf628, buf622, buf619, buf613, buf610, buf604, buf601, buf595, buf592, buf583, buf580, buf574, buf571, buf565, buf562, buf556, buf553, buf547, buf544, buf538, buf535, buf529, buf526, buf520, buf517, buf511, buf508, buf501, buf498, buf492, buf489, buf483, buf480, buf474, buf471, buf465, buf462, buf456, buf453, buf447, buf444, buf438, buf435, buf429, buf426, buf420, buf417, buf410, buf407, buf401, buf398, buf392, buf389, buf383, buf380, buf374, buf371, buf365, buf362, buf356, buf353, buf347, buf344, buf338, buf335, buf329, buf326, buf319, buf316, buf310, buf306, buf300, buf297, buf291, buf288, buf282, buf278, buf272, buf269, buf263, buf260, buf254, buf251, buf245, buf242, buf236, buf232, buf225, buf221, buf215, buf212, buf206, buf203, buf197, buf194, buf188, buf185, buf179, buf176, buf170, buf167, buf158, buf155, buf150, buf146, buf141, buf138, buf132, buf129, buf123, buf120, buf115, buf111, buf106, buf103, buf97, buf94, buf87, buf84, buf78, buf75, buf70, buf66, buf61, buf58, buf52, buf49, buf43, buf40, buf35, buf31, buf26, buf23, buf17, buf14, buf7, buf4, buf879, buf871, buf862, buf851, buf842, buf830, buf820, buf811, buf801, buf792, buf783, buf772, buf762, buf752, buf743, buf733, buf724, buf715, buf704, buf694, buf684, buf675, buf665, buf656, buf647, buf636, buf626, buf617, buf608, buf599, buf587, buf578, buf569, buf560, buf551, buf542, buf533, buf524, buf515, buf505, buf496, buf487, buf478, buf469, buf460, buf451, buf442, buf433, buf424, buf414, buf405, buf396, buf387, buf378, buf369, buf360, buf351, buf342, buf333, buf323, buf314, buf304, buf295, buf286, buf276, buf267, buf258, buf249, buf240, buf229, buf219, buf210, buf201, buf192, buf183, buf174, buf162, buf153, buf145, buf136, buf127, buf118, buf110, buf101, buf91, buf82, buf73, buf65, buf56, buf47, buf38, buf30, buf21, buf11, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((192, 80, 3, 3), (720, 1, 240, 80), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((384, 288, 3, 3), (2592, 1, 864, 288), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((192, 128, 7, 1), (896, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, 128, 7, 1), (896, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, 128, 7, 1), (896, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((192, 128, 1, 7), (896, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((192, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((192, 160, 1, 7), (1120, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((320, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((8, 3, 299, 299), (268203, 1, 897, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 149, 149), (710432, 1, 4768, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 32, 149, 149), (710432, 1, 4768, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 147, 147), (691488, 1, 4704, 32), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 32, 147, 147), (691488, 1, 4704, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 147, 147), (1382976, 1, 9408, 64), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 147, 147), (1382976, 1, 9408, 64), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 64, 73, 73), (341056, 1, 4672, 64), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((8, 64, 73, 73), (341056, 1, 4672, 64), device='cuda:0', dtype=torch.int64)
    convolution_3 = rand_strided((8, 80, 73, 73), (426320, 1, 5840, 80), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 80, 73, 73), (426320, 1, 5840, 80), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 192, 71, 71), (967872, 1, 13632, 192), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 192, 71, 71), (967872, 1, 13632, 192), device='cuda:0', dtype=torch.float32)
    getitem_12 = rand_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cuda:0', dtype=torch.int64)
    convolution_5 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 32, 35, 35), (39200, 1, 1120, 32), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 384, 17, 17), (110976, 1, 6528, 384), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 96, 17, 17), (27744, 1, 1632, 96), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_65 = rand_strided((8, 288, 17, 17), (83232, 1, 4896, 288), device='cuda:0', dtype=torch.int64)
    cat_3 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_3 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_4 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_52 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_54 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_56 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_57 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_5 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_61 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_62 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_64 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_66 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_67 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_205 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_6 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_217 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_72 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_73 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_74 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_159 = rand_strided((8, 768, 8, 8), (49152, 1, 6144, 768), device='cuda:0', dtype=torch.int64)
    cat_8 = rand_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.float32)
    squeeze_229 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_77 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cuda:0', dtype=torch.float32)
    squeeze_241 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_80 = rand_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_81 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_7 = rand_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.float32)
    squeeze_253 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.float32)
    squeeze_256 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_259 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_86 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_262 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_265 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cuda:0', dtype=torch.float32)
    squeeze_268 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_89 = rand_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cuda:0', dtype=torch.float32)
    convolution_90 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_271 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_90 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_274 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.float32)
    squeeze_277 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    avg_pool2d_8 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.float32)
    squeeze_280 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_378 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_402 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_5 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_438 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_450 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bool)
    unsqueeze_474 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_10 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_498 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_510 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_14 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_546 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_15 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bool)
    unsqueeze_582 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_18 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_594 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bool)
    unsqueeze_642 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_24 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_25 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_30 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_738 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_33 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_774 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_34 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_786 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_35 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_798 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_822 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_40 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_858 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_870 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_882 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_43 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_894 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_44 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_906 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_45 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_918 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_930 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_942 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_954 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_966 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_50 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_978 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_990 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1002 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_53 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1014 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_54 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1026 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_55 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1038 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1050 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1062 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1074 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1086 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_60 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1098 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1110 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1122 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_63 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_1134 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_64 = rand_strided((8, 96, 17, 17), (27744, 1, 1632, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_1146 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1158 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1170 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_67 = rand_strided((8, 384, 17, 17), (110976, 1, 6528, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_1182 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_68 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1194 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_69 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_1206 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1218 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1230 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_72 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1242 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1254 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_74 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1266 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_75 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1278 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_76 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_1290 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1302 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1314 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_79 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1326 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1338 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_81 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1350 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_82 = rand_strided((8, 32, 35, 35), (39200, 1, 1120, 32), device='cuda:0', dtype=torch.bool)
    unsqueeze_1362 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_83 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_1374 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1386 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1398 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_86 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1410 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1422 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_88 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda:0', dtype=torch.bool)
    unsqueeze_1434 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1446 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1458 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1470 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1482 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1494 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_567, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_1, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_2, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_3, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_4, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_5, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_6, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, relu_73, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, getitem_159, cat_8, convolution_76, squeeze_229, convolution_77, squeeze_232, relu_77, convolution_78, squeeze_235, convolution_79, squeeze_238, convolution_80, squeeze_241, relu_80, convolution_81, squeeze_244, relu_81, convolution_82, squeeze_247, convolution_83, squeeze_250, avg_pool2d_7, convolution_84, squeeze_253, cat_11, convolution_85, squeeze_256, convolution_86, squeeze_259, relu_86, convolution_87, squeeze_262, convolution_88, squeeze_265, convolution_89, squeeze_268, relu_89, convolution_90, squeeze_271, relu_90, convolution_91, squeeze_274, convolution_92, squeeze_277, avg_pool2d_8, convolution_93, squeeze_280, clone, permute_1, le, unsqueeze_378, le_1, unsqueeze_390, le_2, unsqueeze_402, unsqueeze_414, unsqueeze_426, le_5, unsqueeze_438, le_6, unsqueeze_450, unsqueeze_462, le_8, unsqueeze_474, le_9, unsqueeze_486, le_10, unsqueeze_498, le_11, unsqueeze_510, unsqueeze_522, unsqueeze_534, le_14, unsqueeze_546, le_15, unsqueeze_558, unsqueeze_570, le_17, unsqueeze_582, le_18, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, le_22, unsqueeze_642, unsqueeze_654, le_24, unsqueeze_666, le_25, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, le_30, unsqueeze_738, unsqueeze_750, unsqueeze_762, le_33, unsqueeze_774, le_34, unsqueeze_786, le_35, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_40, unsqueeze_858, unsqueeze_870, unsqueeze_882, le_43, unsqueeze_894, le_44, unsqueeze_906, le_45, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, le_50, unsqueeze_978, unsqueeze_990, unsqueeze_1002, le_53, unsqueeze_1014, le_54, unsqueeze_1026, le_55, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, unsqueeze_1074, unsqueeze_1086, le_60, unsqueeze_1098, unsqueeze_1110, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, unsqueeze_1230, le_72, unsqueeze_1242, unsqueeze_1254, le_74, unsqueeze_1266, le_75, unsqueeze_1278, le_76, unsqueeze_1290, unsqueeze_1302, unsqueeze_1314, le_79, unsqueeze_1326, unsqueeze_1338, le_81, unsqueeze_1350, le_82, unsqueeze_1362, le_83, unsqueeze_1374, unsqueeze_1386, unsqueeze_1398, le_86, unsqueeze_1410, unsqueeze_1422, le_88, unsqueeze_1434, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1482, unsqueeze_1494, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('adv_inception_v3', benchmark_compiled_module)
